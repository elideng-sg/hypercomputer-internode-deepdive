# 22 — Storage & the Data Path: the GPU Is Only as Fast as Its Bytes

## Overview

Parts I–V measured **compute** and **fabric**: how fast a GPU computes, how fast GPUs talk. This document is about the third axis that sets the ceiling on a real job and is invisible in every synthetic benchmark — **how fast bytes reach the GPU**. A training step is `read → H2D copy → compute → (every N steps) checkpoint write`. If the *read* can't keep up, the most expensive H100 in the fleet sits idle at 20% utilization, and the symptom — "training is slow" — looks nothing like its cause.

The lab so far cheated: it fed the GPUs from `torch.randn` in memory, so the data path never existed. [lab-19](../../labs/lab-19-storage-data-path/) removes the cheat — it mounts a **GCS bucket** through the **GCSFuse CSI driver** the way a production JobSet does, and shows the same GEMM step **starved** (reading a fresh shard off the bucket every step) versus **fed** (resident) — a swing in **GPU busy-fraction** from one storage decision.

**What you'll learn:**
- The **storage ladder** for GPU training on GCP — Local SSD → GCSFuse → Filestore/Managed Lustre → Hyperdisk ML — and which symptom each fixes
- Why **GCSFuse + Workload Identity** is the default data path, how it mounts, and where its throughput ceiling bites
- How to *see* a starved GPU: the **io% vs compute% split** and the **GPU busy-fraction** on the same monitoring pipeline (GMP/DCGM) from [doc-20](../part5-operations-diagnostics/20-perf-monitoring-day2.md)
- The **checkpoint** problem — the write half of the data path, and why it's a throughput *and* a topology decision
- The knobs that actually move the number: **file cache**, **parallel downloads**, **sharding**, and **prefetch**

**Prerequisites:** [doc-16](../part5-operations-diagnostics/16-diagnostic-method.md) (read the signal, don't assume) and [doc-20](../part5-operations-diagnostics/20-perf-monitoring-day2.md) (the GMP/DCGM busy-fraction we cross-check against); helpful: [doc-21](21-gke-network-design.md) (WIF and VPC-native are shared prerequisites).

**Instantiated by:** [lab-19](../../labs/lab-19-storage-data-path/) — mount GCS via GCSFuse, measure starved-vs-fed GPU utilization + read throughput + checkpoint write, cross-checked on GMP.

---

## Step 0 — The storage ladder (match the tier to the symptom)

There is no single "GPU storage." Each tier fixes a different symptom; picking the wrong one is a common and expensive mistake.

| Tier | What it is | Fixes | Cost / limit |
|---|---|---|---|
| **Local SSD** | NVMe physically on the A3 node | fastest possible reads; scratch, shard cache | ephemeral (gone on node loss), fixed size |
| **GCSFuse** | GCS bucket mounted as a filesystem (CSI) | huge datasets, checkpoints, shared read — the **default** | object-store latency; throughput ceiling per mount |
| **Filestore / Managed Lustre** | managed NFS / parallel FS | many nodes reading the *same* files at high IOPS | provisioned capacity, price |
| **Hyperdisk ML** | read-only network disk, fan-out to many nodes | serving weights / read-mostly datasets to a fleet | read-only, pre-load step |

**The honest scope of lab-19:** it runs **GCSFuse** (the default and the one that most often starves a job) on the real cluster. Local SSD, Filestore/Lustre, and Hyperdisk ML are described as the ladder and contrasted — the lab measures the rung nearly every GCP GPU job starts on.

---

## Step 1 — GCSFuse + Workload Identity: the default data path

GCSFuse presents a bucket as a POSIX-ish filesystem via a FUSE mount, injected as a **CSI sidecar** into the pod. Two things make it production-shaped rather than a hack:

- **Workload Identity, not keys.** The pod's Kubernetes SA (`lab19-ksa`) is bound to a Google SA (`lab19-gcs@…`) that holds `roles/storage.objectAdmin` on the bucket. No service-account key file ever touches the node — the same WIF mechanism [doc-21](21-gke-network-design.md) calls out as a shared cluster prerequisite. **WIF must be enabled *before* the CSI addon** (lab-19 gotcha G19).
- **A CSI inline volume**, not a manual mount:

```yaml
metadata:
  annotations: { gke-gcsfuse/volumes: "true" }   # inject the sidecar
spec:
  serviceAccountName: lab19-ksa                   # WIF → bucket
  containers:
  - volumeMounts: [{ name: gcs, mountPath: /data }]
  volumes:
  - name: gcs
    csi:
      driver: gcsfuse.csi.storage.gke.io
      volumeAttributes: { bucketName: <bucket>, fileCacheCapacity: "0" }
```

**The throughput ceiling is the whole story.** A single GCSFuse mount reads at a rate set by object-store latency, the number of parallel connections, and MTU — not by the GPU. If a step consumes shards faster than that ceiling, the loader becomes the bottleneck and the GPU idles. lab-19 reads the ceiling directly and measured **~4.9 GiB/s** sequential on the A3 node (gcsfuse 3.11 auto-tunes parallel downloads for the `a3-highgpu-8g` machine type) — so the starve is quantified, not guessed.

> **A node-level gate (why the lab uses userspace GCSFuse).** WIF has *two* switches: the cluster `workloadPool` **and** the node pool's `workloadMetadataConfig=GKE_METADATA`. A pool created before WIF keeps the legacy metadata server, and switching it **recreates the nodes** — unacceptable for a scarce **Flex-start** A3 pool (lab-19 gotcha **G20**, the same class of create/recreate gate as [doc-21](21-gke-network-design.md)'s Dataplane-V2). So lab-19 mounts GCSFuse **in userspace** with a GSA-key Secret (no node changes); the manifest above is the production reference. Same data path, real numbers.

---

## Step 2 — Seeing a starved GPU (the measurement that names the cause)

Same GEMM, same H100. Only the data path differs:

| | STARVED (read every step) | FED (resident) |
|---|---|---|
| where bytes come from | fresh GCSFuse read each step | already in host RAM |
| wall-time split | **high io%**, low compute% | ~0 io%, **high compute%** |
| GPU busy-fraction (NVML) | **low** (idle on I/O) | **high** |
| GMP `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | **low** — confirms it | **high** |
| verdict | *storage-bound* | *compute-bound* |

**Measured live** (lab-19, node `…lq6m`, identical 8192² fp16 GEMM, 40 iters/step): GPU-busy fraction **11.8%** starved vs **100.0%** fed; NVML sampled util 12.1% vs 99.6%; GMP `DCGM_FI_PROF_GR_ENGINE_ACTIVE` plateau **~0.12** vs ramp to **1.000**. Three meters, one story — the starved GPU idles ~88% of the wall. The starve here comes from **arithmetic intensity**: each starved step reads 1 GiB before computing, more than even 4.9 GiB/s storage delivers inside the ~60 ms compute window. A *single* 128 MB read would not starve it — which is the honest nuance (lab-19 gotcha G21).

The value of this is diagnostic discipline. A low `GR_ENGINE_ACTIVE` on a dashboard has **two** causes that look identical — a throttled GPU (lab-17's silent power cap) and a **starved** GPU — and the fix is opposite (fix the device vs fix the loader). lab-19 shows the storage-bound half; you tell them apart by also reading io% and the storage throughput, not just the GPU metric. This is [doc-16](../part5-operations-diagnostics/16-diagnostic-method.md)'s "read the whole path" applied to data.

> **Cross-check, don't trust one meter.** lab-19 reads the busy-fraction two ways — locally via NVML *inside* the step loop, and via **GMP/DCGM** on the managed pipeline — so the storage story survives the [doc-20](../part5-operations-diagnostics/20-perf-monitoring-day2.md) "one tool lies" test.

---

## Step 3 — The checkpoint (the write half)

Reads starve; **writes stall**. Every real job checkpoints its state-dict every N steps, and on a large model that's tens to hundreds of GiB written **through the same data path** — often with all ranks writing at once. Three design consequences:

- **Throughput, again.** A synchronous `torch.save` to GCSFuse blocks the training loop for the write duration — lab-19 measured **87.7 MiB/s** writing a 0.5 GiB state-dict (note: **far below** the 4.9 GiB/s *read* ceiling — GCSFuse writes stage differently), so you can budget the stall. Real jobs overlap it (async / background upload) or write to Local SSD then stage to GCS.
- **Topology / consistency.** GCSFuse is eventually-consistent object storage wearing a filesystem costume; concurrent multi-rank writes to the same prefix need care. Prefer rank-sharded checkpoint files and a single "done" marker.
- **Recovery is a *read* at scale.** Restart pulls the checkpoint back — the read ceiling of Step 1 now gates how fast a preempted job resumes, which ties directly to [doc-19](../part5-operations-diagnostics/19-cluster-job-failure-triage.md) recovery time.

---

## Step 4 — The knobs that move the number

When a job is storage-bound, these are the levers, cheapest first:

- **GCSFuse file cache** (`fileCacheCapacity` > 0, backed by Local SSD): caches shard bytes locally so the *second* epoch reads at SSD speed. lab-19 deliberately sets it to **0** so the starve is visible — in production you turn it **on**.
- **Parallel downloads & bigger reads.** More concurrent connections and larger read sizes raise the per-mount ceiling; jumbo MTU helps here too (the same [doc-21](21-gke-network-design.md) fabric decision).
- **Shard the dataset.** Many mid-sized shards read in parallel beat one giant file or millions of tiny objects (per-object latency dominates the tiny-file case).
- **Prefetch / overlap.** A real `DataLoader` with `num_workers>0` and prefetch hides read latency behind compute — the software fix for the same starve, when the storage tier can't be changed.

The order matters: **measure first** (is io% actually high?), then reach for the cheapest knob that closes the gap.

---

## Key takeaways

- **The GPU is only as fast as its bytes.** A perfectly healthy H100 idles at low utilization if the data path can't feed it — and "training is slow" is the same symptom as a throttled GPU until you read io% and storage throughput.
- **GCSFuse + Workload Identity is the default data path** on GKE — mounted as a CSI sidecar, authenticated without keys; WIF must exist *before* the addon.
- **Diagnose with the split, not the vibe.** io% vs compute% + GPU busy-fraction (NVML **and** GMP/DCGM) names the bottleneck as storage vs compute; the two low-utilization causes have opposite fixes.
- **Checkpoints are the write half** — a throughput budget *and* a consistency/topology decision, and they set recovery time.
- **Match the tier to the symptom** and pull the cheapest knob (file cache → parallelism → sharding → prefetch) that measurement justifies.

---

**Next (Part VI) →** [doc-23 end-to-end training pipeline](23-training-pipeline-jobset.md)
**Builds on →** [doc-16 diagnostic method](../part5-operations-diagnostics/16-diagnostic-method.md) · [doc-20 perf monitoring & day-2 ops](../part5-operations-diagnostics/20-perf-monitoring-day2.md) · [doc-21 GKE network design](21-gke-network-design.md)
**Reference →** [reference-arch-cheatsheet.md](../../reference/reference-arch-cheatsheet.md) · [lab-build-gotchas.md](../../reference/lab-build-gotchas.md)
