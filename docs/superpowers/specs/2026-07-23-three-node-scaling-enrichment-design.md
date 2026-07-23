# Design Spec: 3-node enrichment — *from cliff to curve, from pair to fleet*

**Date:** 2026-07-23 (rev 1)
**Author:** elideng-sg (with Claude Code)
**Status:** Draft — pending user review
**Builds on:** [`2026-07-21-hypercomputer-internode-deepdive-design.md`](2026-07-21-hypercomputer-internode-deepdive-design.md)

---

## 1. Purpose

The guide today is backed by a **2-node** A3 H100 cluster. That proves the single most important fact in the whole repo — the **~17× intra-vs-inter cliff** (~480 GB/s NVLink vs. ~28.6 GB/s TCP) — but it has exactly **two data points** on the node axis (1 node, 2 nodes). Two points prove a *cliff*; they cannot show a *curve*.

A newly-available **3-node** A3 H100 pool (24 × H100) unlocks a class of phenomena that **2 nodes fundamentally cannot measure**. This enrichment adds them — under a single hard rule:

> **North-star rule:** every addition carries an explicit *"why 2 nodes can't show this"* justification. Nothing is added merely because there are more GPUs. If a 2-node pool could already show it, it does not belong in this enrichment.

The deliverable is **"Both"** (per brainstorming): two new flagship labs, one new connective doc, and threaded *third-data-point* updates into the existing labs/docs the new work references.

---

## 2. Live environment (the new capture target)

The 3-node pool is a **different cluster** from the guide's documented lab. This is a first-class methodology fact, not a footnote.

| Property | Guide's existing lab | **New 3-node capture target** |
| :--- | :--- | :--- |
| Cluster | `hypercomputer-a3-cluster` | `hypercomputer-a3-asiaeast1` |
| kube context | `gke_hdlab-elideng_us-central1_hypercomputer-a3-cluster` | `gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1` |
| Region / zone | `us-central1` | `asia-east1-c` |
| GPU pool | `a3-h100-dws-pool` — **2 ×** `a3-highgpu-8g` | `a3-high-flex-pool` — **3 ×** `a3-highgpu-8g` = **24 × H100 80GB** |
| GKE version | `v1.33` | `v1.34` |
| Provisioning | DWS queued, 7-day holds | **Flex-start**, 7-day expiry, capped at 3 (scarce; watchdog re-grabs) |

**Consequences that shape the design:**

1. **Same machine family (A3 High, H100), different cluster.** Driver / NCCL / GKE versions may differ between the two clusters. A scaling curve that mixed clusters would confound *node count* with *software version*. **Therefore the 8 / 16 / 24-GPU scaling curve is captured entirely on `asia-east1-c`** (single cluster, apples-to-apples). The existing `us-central1` numbers are presented separately as a **labeled cross-cluster comparison**, never spliced into the same curve.

2. **Transport is read, not assumed.** As with lab-06, the first captured artifact on the new cluster is the `NCCL_DEBUG=INFO` transport log. If this pool has GPUDirect-TCPX enabled, the numbers will differ from `us-central1`'s TCP floor — that is a **finding to report**, not a discrepancy to hide. The scaling doc adapts its framing to whatever the transport actually is.

3. **Flex nodes are scarce and hard to re-grab.** The node-loss / resilience exercise (lab-13b) **must not drain or delete a real node** — losing scarce Flex capacity would violate the standing "always hold the GPU" posture. Loss is simulated **at the job level** (kill the pods/ranks on one node), which teaches the identical survivor-set lesson while leaving all three nodes held. This mirrors lab-10's existing "all reversible, holder untouched" discipline.

**Cluster facts to characterize during execution (not assumed):** node count and readiness, machine type, `nvidia.com/gpu` allocatable, presence/absence of TCPX/RDMA extended resources and NCCL plugin DaemonSets, actual NCCL transport, and inter-node latency baseline.

---

## 3. What is added

### 3.1 `lab-12` — "Scaling the cliff" (comms & training scaling)

Captures the same measurements at **8 / 16 / 24 GPUs** (1 / 2 / 3 nodes) on `asia-east1-c`.

| Part | Measurement | Why 2 nodes can't show it |
| :--- | :--- | :--- |
| **12a** All-reduce scaling sweep | `busbw` + latency-floor vs. GPU count; capture the `Channel 00/24` ring — the ring now crosses **3** node boundaries (inter-node TCP hops go 2→3) | One inter-node point proves a *cliff*; the *shape* (does peak busbw hold, or degrade with a 3rd slow hop? does the latency floor grow?) needs a 3rd point |
| **12b** Ring vs. Tree | force `NCCL_ALGO=Ring` and `NCCL_ALGO=Tree` at 24 GPUs; sweep small→large messages; find the crossover | At 2 nodes tree≈ring — doc-06 states the algorithm "matters far less than the link"; at 3 nodes tree-depth=2 and the ring/tree divergence is real and measurable |
| **12c** Training scaling | DDP & FSDP step-time at 8 / 16 / 24 GPUs → **strong-scaling efficiency %** and a **weak-scaling** run (grow global batch with node count) | Scaling *efficiency* is a slope; a single 2-node step-time is one point and cannot express it |

### 3.2 `lab-13` — "From pair to fleet" (topology & resilience)

| Part | Measurement | Why 2 nodes can't show it |
| :--- | :--- | :--- |
| **13a** 24-GPU gang across 3 nodes | JobSet + Kueue admitting a **non-power-of-2**, 3-node gang; topology / compact placement; the NCCL ring built across 3 nodes | 2 nodes = exactly one pair: no placement choice, always power-of-2. Ring construction across ≥3 nodes is a distinct, unobservable-at-2 behavior |
| **13b** Node-loss resilience (job-level, Flex-safe) | run a 24-GPU job; kill the ranks on one node; capture the fault signature and demonstrate a **16-GPU survivor set** can reschedule/rerun | Lose 1 of 2 → 0 survivors; there is nothing to reschedule onto. The survivor-set / elastic story requires N≥3 |

### 3.3 `doc-15` — "Scaling: the shape of the cliff" (new connective doc)

A standalone Part-II/III bridge doc that presents, as one narrative: the 8/16/24 **busbw + latency curves**, the **ring/tree crossover**, the **strong/weak scaling-efficiency** story, and the **cross-cluster** comparison caveat. Fed entirely by lab-12 assets. Rationale for standalone (vs. folding into doc-06/09): "scaling" is a distinct concept, the user asked for a flagship, and a single home keeps the 3-point story coherent rather than split across two docs.

### 3.4 Threaded *third-data-point* updates (the in-place half of "Both")

Each existing artifact gets the 24-GPU point added **with an explicit cross-cluster label** so provenance stays honest:

- **doc-06 / lab-06** — add the 24-GPU `busbw` row; reframe "the cliff" → "the cliff *and its shape*"; add the 3-inter-node-hop ring figure; link to doc-15.
- **doc-09 / lab-09** — add 24-GPU DDP/FSDP step-times; add a scaling-efficiency table; link to doc-15/12c.
- **doc-07 / lab-07** and **doc-08 / lab-08** — note 3-node, non-power-of-2 gang placement; link to lab-13a.
- **doc-10 / lab-10** — add the node-loss fault signature to the fault catalog; link to lab-13b.
- **README** — extend the "lab environment" and Part II/III tables; add lab-12, lab-13, doc-15; keep the honesty/status block current (note the second cluster).

---

## 4. Repository changes

```
docs/
  part2-inter-node/06-...           (edit: 24-GPU row, curve framing, link)
  part3-clustering-execution/
    07-..., 08-..., 09-..., 10-...  (edit: 3-node placement / scaling / fault)
  15-scaling-shape-of-the-cliff.md  (NEW connective doc — number 15 appended; both
                                     Part II doc-06 and Part III doc-09 link into it as
                                     the scaling "bridge", avoiding renumber churn)
labs/
  lab-12-scaling-sweep/             (NEW: README, run.sh, reuses allreduce_bench.py)
  lab-13-topology-resilience/       (NEW: README, run.sh)
  lab-06-..., lab-09-...            (edit: 24-GPU third data point + cross-cluster note)
manifests/
  nccl-workbench-c.yaml             (NEW: 3rd 8-GPU workbench pod)
  jobset-nccl-24.yaml               (NEW: 24-GPU / 3-node JobSet gang)
scripts/
  lib_capture.sh                    (edit: parametrize node-pool label + kube context)
  scale_sweep.sh                    (NEW: drives 8/16/24 captures)
  lib_plot.py                       (edit: 3-point scaling-curve + efficiency plots)
assets/
  lab-12/, lab-13/                  (NEW: captured outputs, curves, ring dumps, traces)
VERIFICATION.md                     (append: every asia-east1-c run, cluster explicitly noted)
```

**Tooling change detail:** `lib_capture.sh` currently hardcodes `a3-h100-dws-pool` and the default context. It gains `LAB_NODEPOOL` (default `a3-h100-dws-pool`) and `KUBE_CONTEXT` env overrides so labs can target either cluster without forking the helpers. `cap_verify_provenance` records the **cluster/context** in the note column.

---

## 5. Capture plan (data flow)

1. **Smoke test** — confirm 3 nodes Ready on `asia-east1-c`, GPUs allocatable, cross-node pod-to-pod connectivity, and a 24-GPU rendezvous forms.
2. **Transport first** — capture `NCCL_DEBUG=INFO` on a 24-GPU all-reduce; record transport + the `Channel 00/24` ring. This decides the doc-15 framing.
3. **lab-12a** — all-reduce sweep at 8, 16, 24 GPUs; parse to `busbw`/latency; plot the curve.
4. **lab-12b** — `NCCL_ALGO=Ring`/`Tree` sweeps at 24 GPUs; plot the crossover.
5. **lab-12c** — DDP + FSDP at 8/16/24; strong + weak scaling; parse step-times → efficiency.
6. **lab-13a** — 24-GPU JobSet gang; capture placement, admission, ring.
7. **lab-13b** — job-level node-loss injection; capture fault signature + survivor rerun.
8. Every artifact flows through `lib_capture.sh` → `assets/lab-12|13/` and appends to `VERIFICATION.md` with the cluster named.

---

## 6. Correctness & verification principles (inherited + extended)

- **Never claim a fabric or number not read off the live cluster.** Transport is captured, not assumed.
- **Single-cluster curves.** The 8/16/24 scaling curve is one cluster; cross-cluster numbers are always labeled as such.
- **Flex-safe.** No real node is drained/deleted; resilience is simulated at the job level. All exercises reversible; all three nodes stay held.
- **`busbw = algbw · 2(n-1)/n`** used identically to labs 04/06 so the 24-GPU point is comparable.
- **Provenance** for every run in `VERIFICATION.md`, including cluster/context and node names.

---

## 7. Delivery

- Branch `worktree-3node-enrichment` off `origin/main`; draft PR targeting `main`.
- Build order, each independently shippable: **(1)** lab-12a + doc-06/09 third-point updates + doc-15 skeleton → **(2)** lab-12b + lab-12c → **(3)** lab-13a + lab-13b + doc-07/08/10 updates → **(4)** doc-15 finalized + README.
- Commits follow the repo's existing `lab-NN:` / `doc-NN:` message style.

---

## 8. Explicitly out of scope (YAGNI)

- Enabling GPUDirect-TCPX/TCPXO or RDMA on the new cluster (unless already on — then merely *reported*). Fabric-upgrade work is a separate effort.
- Scaling beyond 3 nodes / 24 GPUs (pool is capped at 3).
- 3D / pipeline-parallel training curricula (PP across nodes) — deferred; DP/FSDP scaling covers the north-star delta.
- Any change to Part IV reference-architecture docs.
- Draining/deleting real nodes for a "true" hardware node-loss test.

---

## 9. Success criteria

1. A reader sees a real **8/16/24-GPU curve** (busbw + latency), not just the two-point cliff.
2. Each of lab-12a/b/c and lab-13a/b states — and its captured data demonstrates — a phenomenon a **2-node pool cannot show**.
3. All 24-GPU numbers are traceable in `VERIFICATION.md` to `asia-east1-c`, with transport read off the wire.
4. Cross-cluster comparisons are labeled; no curve mixes clusters.
5. All three Flex nodes remain held throughout; every exercise is reversible.
6. New docs/labs match the repo's existing rigor, tone, mermaid-figure style, and cross-linking.
