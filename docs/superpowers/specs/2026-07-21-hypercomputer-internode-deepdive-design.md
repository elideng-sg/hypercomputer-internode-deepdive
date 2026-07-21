# Design Spec: `hypercomputer-internode-deepdive`

**Date:** 2026-07-21
**Author:** elideng-sg (with Claude Code)
**Status:** Draft — awaiting user review

---

## 1. Purpose

A **standalone, comprehensive, hands-on guide** to two tightly-coupled topics on Google Cloud AI Hypercomputer:

1. **Inter-node communication** — how GPUs across separate machines actually exchange data: NVLink/NVSwitch inside a node, NICs / RDMA / GPUDirect between nodes, and NCCL collectives on top.
2. **Distributed workload execution** — how a multi-node GPU job is scheduled, launched, synchronized, and observed: GKE device plugin & topology, JobSet/Kueue gang scheduling, `torchrun` rendezvous, DDP/FSDP.

Every mechanism layer is paired with a **hands-on lab executed on a live 2-node A3 H100 cluster**, capturing **real measured outputs** (bandwidths, topology dumps, NCCL traces, failure signatures) rather than generic expected values.

The guide is **self-contained**: it re-explains the foundations needed so a reader never has to leave, while remaining focused on the inter-node / execution story.

### Target reader
An engineer or data scientist who can use `kubectl` and Python but wants to build **deep, first-principles knowledge** of how GPU workloads communicate and execute on GCP — enough to **design, implement, troubleshoot, and debug** them. Depth target matches the existing `gcp-ai-infra-study` notes (textbook rigor + practitioner detail).

---

## 2. Live environment (the lab)

| Property | Value |
| :--- | :--- |
| Project | `hdlab-elideng` |
| Cluster | `hypercomputer-a3-cluster` (GKE `v1.33`) |
| Region | `us-central1` |
| GPU pool | `a3-h100-dws-pool` — **2 × `a3-highgpu-8g`** = **16 × H100 80GB** |
| Provisioning | Dynamic Workload Scheduler (DWS) queued provisioning, 7-day holds |
| Aux pool | `default-pool` (e2-standard-4) for control/CPU workloads |

**Verified cluster facts to characterize during execution (not assumed):**
- Nodes currently expose only `nvidia.com/gpu: 8` as an extended resource — **no dedicated GPU-NIC / GPUDirect-TCPX extended resources are present**.
- `nccl-fastsocket-installer` DaemonSet exists but has **0 pods scheduled**; **no `tcpx`/`tcpxo` DaemonSet** is installed.
- **Implication:** as provisioned, inter-node NCCL traffic most likely traverses the standard gVNIC/VPC TCP path rather than GPUDirect. **Characterizing this actual data path — and, where feasible, enabling GPUDirect-TCPX / NCCL Fast Socket and benchmarking the before/after delta — is a core teaching thread of the guide**, not an assumption to paper over.

A machine-generated `VERIFICATION.md` will record exactly what was run, when, on which nodes, and which artifact resulted, so every documented number is traceable and reproducible.

---

## 3. Repository architecture

```
hypercomputer-internode-deepdive/
├── README.md                 # Guide map, cluster-at-a-glance, reading order, how to run labs
├── docs/                     # Mechanism deep-dives — one file per layer
│   ├── 00-overview-topology.md
│   ├── 01-intranode-nvlink-nvswitch.md
│   ├── 02-nic-rdma-gpudirect.md
│   ├── 03-nccl-collectives.md
│   ├── 04-gke-scheduling-topology.md
│   ├── 05-job-frameworks-jobset-kueue.md
│   ├── 06-distributed-training-ddp-fsdp.md
│   └── 07-observability-debugging.md
├── labs/                     # Step-by-step practice; one dir per lab
│   ├── lab-01-verify-topology/
│   ├── lab-02-nccl-tests-bandwidth/
│   ├── lab-03-network-path-tuning/
│   ├── lab-04-2node-pytorch-ddp/
│   ├── lab-05-jobset-multinode/
│   └── lab-06-fault-injection-debug/
├── manifests/                # Reusable, verified K8s YAML (JobSet, PyTorchJob, DaemonSets, ConfigMaps)
├── scripts/                  # Runner + capture/parse scripts (collect NCCL/DCGM/topology → CSV/plots)
├── assets/                   # Captured REAL outputs: logs, CSVs, rendered plots, topology diagrams
├── reference/                # Env-var cheat sheets, XID table, NCCL tunables, glossary
└── VERIFICATION.md           # Provenance log of every live run and artifact
```

**Layout contract for each `labs/lab-XX/`:**
- `README.md` — objective, prerequisites, numbered steps, **real captured output**, and interpretation ("what this number means / what good vs bad looks like").
- The manifest(s) and `run.sh` it uses (or references into `manifests/`).
- Links back to the `docs/` layer whose mechanism it exercises.

**Doc ↔ lab pairing:** each `docs/NN-*.md` ends with a "Practice" section linking to its lab; each lab links back up to its mechanism doc. This is the two-way spine.

---

## 4. Content spine (layered bottom-up)

Each layer = **mechanism doc + matching lab**. Ordered from silicon upward.

| # | Layer | Mechanism doc covers | Lab (live) |
| :-- | :--- | :--- | :--- |
| 0 | **Topology overview** | The 16×H100 layout; A3 machine families (High/Mega/Ultra) and their networking (TCPX / TCPXO / RDMA-RoCE); what is physically wired vs. what is configured here | `lab-01`: enumerate the real cluster — nodes, GPUs, NICs, extended resources, DaemonSets |
| 1 | **Intra-node: NVLink / NVSwitch** | Gen4 NVLink, NVSwitch, ~900 GB/s bisection; why intra-node bandwidth ≫ inter-node; NVLink vs PCIe paths | `lab-01` cont.: `nvidia-smi topo -m`, per-link NVLink bandwidth on one node |
| 2 | **Inter-node: NICs, RDMA, GPUDirect** | gVNIC vs. dedicated GPU NICs; RDMA/RoCE vs. GPUDirect-TCPX vs. TCPXO; the NCCL network plugin DaemonSet model; `NCCL_*` / `NCCL_GPUDIRECTTCPX_*` env; **the actual data path on this cluster** | `lab-02`/`lab-03`: inspect NICs & plugin path; identify current path; (if feasible) deploy Fast Socket / TCPX plugin |
| 3 | **NCCL collectives** | All-reduce/all-gather/reduce-scatter mechanics; ring vs. tree algorithms; topology discovery; message-size vs. bandwidth curve; bus vs. algo bandwidth | `lab-02`: `nccl-tests` all-reduce & all-gather across 2 nodes — **thorough message-size sweep**, ring vs tree, → CSV → plotted GB/s curves |
| 4 | **GKE scheduling & topology** | Device plugin; GPU requests/limits & tolerations; topology-aware scheduling; gang requirements; DWS queued provisioning & capacity holders | `lab-05` foundation: correctly schedule a 16-GPU gang across both nodes |
| 5 | **Job frameworks** | JobSet (replicated jobs) + Kueue (queueing/quota); headless service & pod DNS for rendezvous; why gang scheduling matters | `lab-05`: multi-node JobSet running a distributed workload across 16 GPUs |
| 6 | **Distributed training** | DDP vs. FSDP; gradient bucketing → which collective; `torchrun`/`torchx` rendezvous; how a training step maps onto NCCL calls | `lab-04`: 2-node / 16-GPU PyTorch job with `NCCL_DEBUG` tracing; DDP and FSDP compared |
| 7 | **Observability & debugging** | DCGM metrics; `NCCL_DEBUG`/`NCCL_DEBUG_SUBSYS`; XID error taxonomy; `ethtool`/NIC counters; diagnosing slow, stalled, or mismatched collectives; hangs & timeouts | `lab-06`: inject faults (kill a rank, saturate a NIC, force a version mismatch) and read the signatures |

Message-size sweeps and ring/tree comparisons are run **thoroughly** (repeated runs for stable numbers, multiple algorithms), making best use of the held GPUs.

---

## 5. Live-lab data flow

For each lab:

```
run.sh → submit workload (kubectl/JobSet/PyTorchJob)
       → scripts/ capture NCCL logs + DCGM + topology + timings
       → parse to CSV (scripts/parse_*.py)
       → render plots/tables → assets/
       → lab README embeds the ACTUAL numbers + interpretation
       → VERIFICATION.md appends provenance (what/when/where/artifact)
```

Design consequence: the guide reflects **this** cluster's measured behavior, and re-running a lab regenerates its artifacts deterministically. Long-running sweeps keep the GPUs busy, consistent with the standing "never leave the GPU idle" practice.

---

## 6. Correctness & verification principles

1. **Execute before documenting.** Every command, manifest, and number is run on the live cluster first; captured output is embedded rather than invented.
2. **Verify hardware claims against running nodes.** Machine-family networking (TCPX vs. TCPXO vs. RDMA), NIC count/type, NVLink generation, and the actual NCCL data path are confirmed on the nodes — never assumed from marketing docs.
3. **Traceable provenance.** `VERIFICATION.md` links every documented artifact to the run that produced it.
4. **Reproducibility.** Each lab's `run.sh` + manifests reproduce its artifacts on the same cluster.
5. **Honest gaps.** If a capability (e.g., full GPUDirect-TCPX) cannot be enabled in this environment, the guide says so explicitly and documents the standard-path behavior it did measure, plus what would change with the feature enabled.

---

## 7. Delivery

- New **public** GitHub repo `elideng-sg/hypercomputer-internode-deepdive`.
- Work isolated in a git worktree; incremental, clean commits per module (doc + lab together where possible).
- Fresh repo (no branch protection) → push to `main`; no PR gate needed for the initial build, though module commits stay reviewable.
- Cluster mutations (deploying plugins, submitting jobs) are additive and reversible; teardown/cleanup steps are documented and capacity holders are preserved so GPU holds are never dropped.

---

## 8. Explicitly out of scope (YAGNI)

- TPU communication (ICI/OCS) — this guide is GPU/NCCL focused; mentioned only for contrast.
- Single-node-only tutorials — covered elsewhere; here only as the intra-node baseline for the inter-node story.
- Deep GPU microarchitecture and driver-install troubleshooting — already covered by the existing `gcp-ai-infra-study` and `gpu_driver_troubleshooting_guide` (linked, not duplicated, despite the standalone goal — foundations are re-explained, but exhaustive driver-branch matrices are referenced).
- Model/accuracy quality of the training workloads — jobs exist to generate communication and execution behavior, not to train a useful model.

---

## 9. Success criteria

- A reader can follow the guide top-to-bottom and, on an equivalent A3 cluster, reproduce every lab.
- The guide explains **why** each number is what it is (mechanism → measurement), not just how to run a command.
- Real captured data (bandwidth curves, topology, traces, failure signatures) is present for every lab that ran.
- Troubleshooting/debugging content maps concrete symptoms → root causes → fixes, backed by at least one deliberately injected real failure.
