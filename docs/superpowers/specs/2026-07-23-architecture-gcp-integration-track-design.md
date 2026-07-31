# Design Spec: Architecture & GCP Integration use-cases track (Part VI)

**Date:** 2026-07-23 (rev 1)
**Author:** elideng-sg (with Claude Code)
**Status:** Draft — pending user review
**Sibling specs:** [`…-three-node-scaling-enrichment-design.md`](2026-07-23-three-node-scaling-enrichment-design.md) · [`…-diagnostics-troubleshooting-operations-track-design.md`](2026-07-23-diagnostics-troubleshooting-operations-track-design.md) (all three independent; ship separately)
**Parent design:** [`2026-07-21-hypercomputer-internode-deepdive-design.md`](2026-07-21-hypercomputer-internode-deepdive-design.md)

---

## 1. Purpose & motivation

The guide teaches the GPU stack **vertically** (silicon → NVLink → NCCL → cluster → platform) with great rigor, but it is thin on the **horizontal** axis engineers most need for real work: **how a GPU workload is actually architected and deployed on GCP** — the *design* skill. Two concrete gaps:

1. **GKE network design is never taught as a design decision.** doc-05 names the GPUDirect-TCPX/TCPXO ladder, but the guide only ever runs the **single-gVNIC** path and honestly reports the ~28.6 GB/s TCP floor. The *architecture that closes that cliff* — multi-network node pools, `GKENetworkParamSet`/`Network` CRDs, jumbo MTU, the NCCL TCPX plugin — is absent. **Verified:** both clusters are single-gVNIC, no multi-network CRDs, `nccl-fastsocket-installer` at 0 pods, no TCPX/TCPXO/RDMA DaemonSets.
2. **Integration with common GCP services is absent.** Real workloads live inside an architecture: data in GCS/Parallelstore, checkpoints to GCS, images from Artifact Registry, identity via Workload Identity Federation, metrics in Managed Prometheus, serving behind a gateway. None of this end-to-end wiring appears.

This track adds a **Part VI — Architecture & GCP Integration** as **reference use-cases**: each is a realistic scenario an engineer must *design and stand up*, not just observe. It covers the four areas the user selected, and — per the user's explicit choice — includes a **live GPUDirect-TCPX before/after**, provisioning a real multi-network pool to **measure the cliff actually closing**, rather than only describing it.

**Guiding rules (inherited & sharpened):**
- **Design-first:** every lab hands the engineer an architecture goal and the design tradeoffs, then stands it up live.
- **Measured, not asserted:** the network-design payoff is captured (before/after NCCL numbers, transport read off the wire); knowledge-first pieces (RDMA/A3 Ultra, TCPXO/A3 Mega if unavailable) are clearly labeled reference-architecture, never claimed as run.
- **Both lenses:** the GCP architecture view (VPC/CRDs/IAM/services) and the NVIDIA view (NCCL transport, DCGM) side by side.
- **Hold-safe:** the new TCPX pool respects the cap-of-3 A3-Flex stop-line and does not disturb existing holders; provisioning and teardown are documented and reversible.

---

## 2. New live environment (the TCPX workstream)

To demonstrate the *enabled* network design, a **new GPUDirect-TCPX-capable A3 High node pool** is provisioned (the multi-network design is fixed at pool-creation, so it cannot be toggled on an existing pool).

| Property | Value / plan |
| :--- | :--- |
| Machine | `a3-highgpu-8g` (H100) — matches existing hardware; TCPX (4 GPU NICs) |
| Networking | 4 additional GPU VPCs + subnets; `Network` + `GKENetworkParamSet` CRDs; jumbo MTU 8244/8896 |
| NCCL plugin | `nccl-tcpx-installer` DaemonSet + `tcpx-daemon`; workloads annotated with the GPU networks and TCPX env |
| Size | ≥2 nodes (16 GPUs) to show inter-node; provisioned under the A3-Flex cap-of-3 stop-line |
| Cluster | reuse an existing A3 cluster (add pool) or a dedicated cluster in a region with A3 capacity |

**Reference rungs kept knowledge-first (not run unless hardware appears):** TCPXO (A3 Mega, `a3-megagpu-8g`, 8 NICs — higher ceiling); GPUDirect-RDMA/RoCE (A3 Ultra `a3-ultragpu-8g` / A4, CX-7, `NET/IB` present). These are documented as the design ladder and contrasted with the measured TCPX result.

**Cross-track tie:** once TCPX is live, the scaling track's 8/16/24-GPU sweep is re-run on it to produce the **enabled scaling curve** — the two tracks reinforce each other. Numbers are always labeled by pool/fabric.

---

## 3. What is added

### Part VI — Architecture & GCP Integration (new doc set)

| Doc | Title | Role |
| :--- | :--- | :--- |
| **doc-21** | GKE network design for GPU workloads | Single-gVNIC vs TCPX vs TCPXO vs RDMA as a **design decision**; multi-network CRDs, jumbo MTU, Dataplane V2, VPC-native, private cluster + Cloud NAT, Shared VPC, compact/topology placement & rail alignment, serving networking (Gateway API / Inference Gateway). Tied to the measured cliff. |
| **doc-22** | Storage & the data path | GCSFuse CSI, Parallelstore, Managed Lustre, Hyperdisk ML, Filestore; checkpoint I/O to GCS; the "GPUs starved by the dataloader" diagnostic. Design tradeoffs (throughput vs cost vs POSIX). |
| **doc-23** | End-to-end training pipeline (reference architecture) | Data → JobSet training → checkpoints → metrics, with IAM/WIF + Artifact Registry wiring. Ties Parts II/III/V together. |
| **doc-24** | Inference serving & MLOps | Online serving, GKE Inference Gateway, autoscaling on GPU/DCGM metrics, Vertex AI vs self-managed GKE contrast, image supply chain (Artifact Registry, Cloud Build). |

### Labs (each = design a scenario, stand it up, measure)

| Lab | Scenario / design goal | Live? | Tools & GCP services exercised |
| :--- | :--- | :--- | :--- |
| **lab-18** Enable GPUDirect-TCPX (flagship) | "Close the cliff": design & provision the multi-network TCPX pool, install the plugin, re-run the NCCL all-reduce, capture **before (gVNIC ~28 GB/s) → after (TCPX)**; read the changed transport (`NET/GPUDirectTCPX`, GPU Direct RDMA/DMA lines) | **Live** | gcloud pool + additional networks, `Network`/`GKENetworkParamSet`, `nccl-tcpx-installer`, NCCL env, `all_reduce_perf`/torch bench, `ethtool`/MTU |
| **lab-19** Storage & data-path throughput | "Feed the GPUs": compare GCSFuse vs Parallelstore/Hyperdisk ML read throughput; a **dataloader-bound** training step vs storage-optimized; checkpoint write to GCS; diagnose a **starved-GPU** (low SM occupancy, high wait) | **Live** | GCSFuse CSI, Parallelstore/Hyperdisk ML CSI, `fio`, DCGM `DCGM_FI_PROF_*`, PyTorch DataLoader, GCS |
| **lab-20** End-to-end training pipeline | Design & run the full reference use-case: data in GCS/Parallelstore → **JobSet** distributed training → **checkpoints to GCS** → **Managed Prometheus/Grafana** metrics; wire **WIF** + **Artifact Registry** | **Live** | JobSet/Kueue, WIF, Artifact Registry, GMP, GCS, (build via Cloud Build) |
| **lab-21** Inference serving + autoscaling | Deploy a GPU model server; front it with **GKE Inference Gateway** / L7 LB; **autoscale on DCGM GPU utilization** (HPA/custom metrics); load-test; contrast **Vertex AI** managed serving | **Live** (Vertex contrast read-only) | Inference Gateway/Gateway API, HPA + custom-metrics adapter, DCGM metrics, load generator, Vertex AI |

### Cross-cutting (woven through labs 20–21, documented once)
Workload Identity Federation, Artifact Registry image supply chain, Cloud Build CI, Secret Manager, and the IAM model for pod→GCS/AR access.

---

## 4. Repository changes

```
docs/
  part6-architecture-gcp-integration/
    21-gke-network-design.md            (NEW — hub for the network-design decision)
    22-storage-and-data-path.md         (NEW)
    23-end-to-end-training-pipeline.md  (NEW)
    24-inference-serving-mlops.md       (NEW)
labs/
  lab-18-enable-gpudirect-tcpx/         (NEW — flagship, live before/after)
  lab-19-storage-data-path/             (NEW)
  lab-20-e2e-training-pipeline/         (NEW)
  lab-21-inference-serving-autoscale/   (NEW)
manifests/
  tcpx/ (network-crds.yaml, nccl-tcpx-installer.yaml, workbench-tcpx.yaml),
  storage/ (gcsfuse-pvc.yaml, parallelstore-pvc.yaml),
  serving/ (model-server.yaml, inference-gateway.yaml, hpa-dcgm.yaml)   (NEW)
scripts/
  provision_tcpx_pool.sh  (NEW — documented, reversible pool + networks setup),
  lib_capture.sh          (reuse; already context/pool-parametrized by sibling spec)
docs/00-guide-overview.md, README.md   (edit: add Part VI + labs 18-21; update env table with TCPX pool)
docs/part2-inter-node/05-*, 06-*       (edit: link to lab-18 as the "enabled" counterpart to the TCP floor)
reference/reference-arch-cheatsheet.md (edit: add GCP integration reference patterns)
VERIFICATION.md                        (append: TCPX provisioning + before/after captures)
```

---

## 5. Provisioning & capture plan

**TCPX workstream (lab-18) — DELIVERED 2026-07-31, reversible.** All five steps below were executed; step 4's `after` is measured (`Using network GPUDirectTCPX_v7`, 0× `NET/Socket`, **83.27 GB/s** busbw on 16×H100). Step 2 needed a **new cluster** — multi-networking requires Dataplane V2 and both are create-time-only (**G17**) — and step 3 needed the plugin pinned to `:v3.1.12` (**G29**) and the `pause` image re-pinned (**G27**). The original plan text follows:
1. Create 4 GPU VPCs + subnets (jumbo MTU) — or reuse if present.
2. Create the A3 High node pool with `--additional-node-network` ×4 (+ pod networks); apply `Network`/`GKENetworkParamSet`.
3. Install `nccl-tcpx-installer`; verify DaemonSet Ready and NIC resources on nodes.
4. Capture **before** = existing gVNIC NCCL result (already have it); capture **after** = TCPX-annotated workload, reading `NCCL_DEBUG` transport + busbw sweep.
5. Document teardown; keep the pool held per the stop-line if kept for the scaling cross-run.

**Other labs:** each captures a healthy baseline + the design artifacts (CRDs, PVCs, gateway, HPA), plus a measured result (storage throughput, pipeline run, autoscale response). All via `lib_capture.sh` → `assets/` + `VERIFICATION.md`, pool/cluster labeled.

**Build order (each independently shippable):**
1. **lab-18 TCPX** + doc-21 (flagship; the measured cliff-closing) — highest value, and unblocks the enabled scaling cross-run.
2. **lab-19 storage/data-path** + doc-22.
3. **lab-20 e2e training pipeline** + doc-23 (integrates 18/19).
4. **lab-21 inference serving** + doc-24.

---

## 6. Correctness & verification principles (inherited)

- Never claim a fabric/number not read off a live run; TCPX before/after is measured with transport read off `NCCL_DEBUG`.
- Knowledge-first rungs (TCPXO/RDMA) labeled as reference-architecture, not run.
- Hold-safe: new pool respects the A3-Flex cap-of-3 stop-line; existing holders untouched; provisioning reversible with documented teardown.
- Both-lens honesty; full provenance in `VERIFICATION.md` (pool/cluster named).

---

## 7. Explicitly out of scope (YAGNI)

- RDMA/RoCE **live** demos (need A3 Ultra / A4 hardware) — reference-architecture only unless that hardware is provisioned.
  - **Superseded 2026-07-28 for TCPXO:** the "unless that hardware is provisioned" condition
    was met. A3 Mega Flex capacity was obtained and `hypercomputer-a3-tcpxo` was built with
    the create-time gates, 8 GPU VPCs, the CRDs and the NCCL plugin — all applied live, with
    8 GPU NICs realised on-node. TCPXO is therefore **in scope** and covered by lab-22 +
    doc-25. RDMA/RoCE remains out of scope. Note the residual limit: fabric *configuration*
    is verified; an enabled-fabric *throughput* number still awaits concurrent Flex capacity.
- Multi-cluster / multi-region fleet networking beyond a design discussion.
- Full production MLOps platform (experiment tracking, feature stores, model registry governance) — reference the pattern, don't build it.
- Cost-optimization / FinOps tooling beyond noting tradeoffs.
- Rewriting existing labs (only cross-links + the env-table update).

---

## 8. Success criteria

1. **lab-18 shows the cliff closing with real numbers** — a measured gVNIC→TCPX before/after with the transport line changing from `NET/Socket` to the TCPX path.
2. Each of the four areas has a **live, design-first** lab where the engineer stands up an architecture and measures a result (not healthy-path observation).
3. The **GCP-integration wiring** engineers actually need — WIF, Artifact Registry, GCSFuse/Parallelstore, GMP, Inference Gateway, HPA-on-DCGM — is demonstrated end to end.
4. The network-design **decision** is teachable: single-gVNIC vs TCPX vs TCPXO vs RDMA, each attributed to its A3 family and tied to measured or referenced bandwidth.
5. The TCPX pool is provisioned hold-safe and its teardown documented; all provenance recorded.
6. New Part VI matches the repo's rigor, tone, mermaid style, and cross-linking; links into Parts II/III/V where the use-cases converge.
