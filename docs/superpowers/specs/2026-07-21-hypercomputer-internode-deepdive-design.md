# Design Spec: `hypercomputer-internode-deepdive`

**Date:** 2026-07-21 (rev 2)
**Author:** elideng-sg (with Claude Code)
**Status:** Draft — awaiting user review

---

## 1. Purpose

A **standalone, comprehensive, hands-on guide** that walks an engineer all the way from a **single GPU** to a **multi-node GPU cluster** on Google Cloud AI Hypercomputer, teaching both the **mechanism** (why it works) and the **practice** (how to run it), with every layer backed by a **lab executed live on a real 2-node A3 H100 cluster**.

The journey is deliberately continuous and bottom-up:

1. **Single node** — how one GPU is built and executes work (microarchitecture), how its drivers/CUDA stack is installed and troubleshot, how a single-GPU workload runs, and how the 8 GPUs inside one node talk over NVLink/NVSwitch.
2. **Inter-node communication** — how GPUs across separate machines exchange data: NICs, RDMA, GPUDirect, and NCCL collectives.
3. **Clustering & distributed execution** — how a multi-node GPU job is scheduled, gang-launched, synchronized, tuned, and debugged: GKE device plugin & topology, JobSet/Kueue, `torchrun` rendezvous, DDP/FSDP, and observability.

Every mechanism layer is paired with a **hands-on lab** that captures **real measured outputs** (bandwidths, topology dumps, NCCL traces, XID/failure signatures) rather than generic expected values.

The guide is **fully self-contained**: GPU microarchitecture and driver/CUDA troubleshooting are taught **in full here**, not merely referenced — so a reader can go from zero to debugging a 16-GPU distributed job without leaving the repo.

### Target reader
An engineer or data scientist who can use `kubectl` and Python and wants to build **deep, first-principles knowledge** of how GPU workloads are built, communicate, and execute on GCP — enough to **design, implement, troubleshoot, and debug** them from a single card to a cluster. Depth target matches the existing `gcp-ai-infra-study` notes (textbook rigor + practitioner detail).

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
├── docs/                     # Mechanism deep-dives — one file per layer, grouped in 3 parts
│   ├── part1-single-node/
│   │   ├── 00-overview-and-cluster.md
│   │   ├── 01-gpu-microarchitecture.md
│   │   ├── 02-drivers-cuda-install-troubleshooting.md
│   │   ├── 03-single-gpu-execution.md
│   │   └── 04-intranode-nvlink-nvswitch.md
│   ├── part2-inter-node/
│   │   ├── 05-nic-rdma-gpudirect.md
│   │   └── 06-nccl-collectives.md
│   └── part3-clustering-execution/
│       ├── 07-gke-scheduling-topology.md
│       ├── 08-job-frameworks-jobset-kueue.md
│       ├── 09-distributed-training-ddp-fsdp.md
│       └── 10-observability-debugging.md
├── labs/                     # Step-by-step practice; one dir per lab, with real captured output
│   ├── lab-01-gpu-arch-inspect/
│   ├── lab-02-driver-cuda-verify/
│   ├── lab-03-single-gpu-benchmark/
│   ├── lab-04-intranode-nvlink/
│   ├── lab-05-network-path-inspect/
│   ├── lab-06-nccl-tests-internode/
│   ├── lab-07-gke-gang-schedule/
│   ├── lab-08-jobset-multinode/
│   ├── lab-09-2node-pytorch-ddp-fsdp/
│   └── lab-10-fault-injection-debug/
├── manifests/                # Reusable, verified K8s YAML (JobSet, PyTorchJob, DaemonSets, ConfigMaps)
├── scripts/                  # Runner + capture/parse scripts (collect NCCL/DCGM/topology → CSV/plots)
├── assets/                   # Captured REAL outputs: logs, CSVs, rendered plots, topology diagrams
├── reference/                # Env-var cheat sheets, XID table, NCCL tunables, driver matrix, glossary
└── VERIFICATION.md           # Provenance log of every live run and artifact
```

**Layout contract for each `labs/lab-XX/`:**
- `README.md` — objective, prerequisites, numbered steps, **real captured output**, and interpretation ("what this number means / what good vs bad looks like").
- The manifest(s) and `run.sh` it uses (or references into `manifests/`).
- Links back to the `docs/` layer whose mechanism it exercises.

**Doc ↔ lab pairing:** each `docs/**/NN-*.md` ends with a "Practice" section linking to its lab; each lab links back up to its mechanism doc. This is the two-way spine.

---

## 4. Content spine (single node → inter-node → cluster)

Each layer = **mechanism doc + matching lab**, ordered as one continuous bottom-up journey.

### Part I — Single Node

| # | Layer | Mechanism doc covers | Lab (live) |
| :-- | :--- | :--- | :--- |
| 0 | **Overview & the cluster** | What the guide teaches and in what order; the 16×H100 environment; A3 machine families (High/Mega/Ultra) and their networking (TCPX / TCPXO / RDMA-RoCE); how to run the labs safely (DWS holds preserved) | `lab-01` intro: enumerate the real cluster — nodes, GPUs, NICs, extended resources, DaemonSets |
| 1 | **GPU microarchitecture** | CPU vs GPU vs TPU design philosophy; SM / SIMT / warps / branch divergence; memory hierarchy (registers → shared → L2 → HBM); Tensor Cores & FP8/BF16; H100 (Hopper) specifics — HBM3 bandwidth, TMA, thread-block clusters | `lab-01`: `nvidia-smi`, `deviceQuery`, inspect SM count / clocks / Tensor Core support on a real H100 |
| 2 | **Drivers, CUDA & troubleshooting** | NVIDIA driver branches (LTSB/PB/NFB) & GCP matrix; GKE managed driver install (device plugin DaemonSets) vs GCE manual install; CUDA toolkit/runtime vs driver; XID error taxonomy; thermal throttling; diagnostics workflow | `lab-02`: verify driver/CUDA versions, read `dmesg`/XID + DCGM diagnostics, interpret a real (or induced) fault |
| 3 | **Single-GPU execution** | The CUDA execution model end to end: host→device, streams, kernels, occupancy, roofline; how a single-GPU training/inference step actually runs; measuring compute vs memory bound | `lab-03`: single-GPU GEMM / FP8 microbench + memory-bandwidth test → measured TFLOPs & GB/s, roofline placement |
| 4 | **Intra-node: NVLink / NVSwitch** | The 8-GPU node fabric: gen4 NVLink, NVSwitch, ~900 GB/s bisection; NVLink vs PCIe paths; why intra-node bandwidth ≫ inter-node — the key motivation for everything in Parts II–III | `lab-04`: `nvidia-smi topo -m`; single-node 8-GPU NCCL all-reduce; per-link NVLink bandwidth → measured |

### Part II — Inter-node Communication

| # | Layer | Mechanism doc covers | Lab (live) |
| :-- | :--- | :--- | :--- |
| 5 | **NICs, RDMA, GPUDirect** | gVNIC vs dedicated GPU NICs; RDMA/RoCE vs GPUDirect-TCPX vs TCPXO; the NCCL network-plugin DaemonSet model; `NCCL_*` / `NCCL_GPUDIRECTTCPX_*` env; **the actual data path on this cluster** and how to characterize it | `lab-05`: inspect NICs & plugin path; identify current path; (if feasible) deploy Fast Socket / TCPX plugin and prepare A/B |
| 6 | **NCCL collectives** | All-reduce/all-gather/reduce-scatter mechanics; ring vs tree algorithms; topology discovery; message-size vs bandwidth curve; bus vs algo bandwidth; how the network path changes the curve | `lab-06`: `nccl-tests` all-reduce & all-gather across 2 nodes — **thorough message-size sweep**, ring vs tree, standard-path vs plugin → CSV → plotted GB/s curves |

### Part III — Clustering & Distributed Execution

| # | Layer | Mechanism doc covers | Lab (live) |
| :-- | :--- | :--- | :--- |
| 7 | **GKE scheduling & topology** | Device plugin; GPU requests/limits & tolerations; topology-aware scheduling; gang requirements; DWS queued provisioning & capacity holders | `lab-07`: correctly schedule a 16-GPU gang across both nodes |
| 8 | **Job frameworks** | JobSet (replicated jobs) + Kueue (queueing/quota); headless service & pod DNS for rendezvous; why gang scheduling matters | `lab-08`: multi-node JobSet running a distributed workload across 16 GPUs |
| 9 | **Distributed training** | DDP vs FSDP; gradient bucketing → which collective; `torchrun`/`torchx` rendezvous; how a training step maps onto NCCL calls; scaling efficiency | `lab-09`: 2-node / 16-GPU PyTorch job with `NCCL_DEBUG` tracing; DDP and FSDP compared |
| 10 | **Observability & debugging** | DCGM metrics; `NCCL_DEBUG`/`NCCL_DEBUG_SUBSYS`; XID revisited at cluster scale; `ethtool`/NIC counters; diagnosing slow, stalled, or mismatched collectives; hangs & timeouts | `lab-10`: inject faults (kill a rank, saturate a NIC, force a version mismatch) and read the signatures |

Single-GPU and single-node microbenchmarks (Part I) establish the **baseline numbers** that make the inter-node and cluster measurements (Parts II–III) meaningful — e.g., NVLink intra-node bandwidth vs. inter-node NCCL bus bandwidth. Message-size sweeps and ring/tree comparisons are run **thoroughly** (repeated runs for stable numbers, multiple algorithms), making best use of the held GPUs.

---

## 5. Live-lab data flow

For each lab:

```
run.sh → submit workload (nvidia-smi / CUDA microbench / kubectl / JobSet / PyTorchJob)
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
2. **Verify hardware claims against running nodes.** Machine-family networking (TCPX vs TCPXO vs RDMA), NIC count/type, NVLink generation, SM/clock/Tensor-Core specs, and the actual NCCL data path are confirmed on the nodes — never assumed from marketing docs.
3. **Traceable provenance.** `VERIFICATION.md` links every documented artifact to the run that produced it.
4. **Reproducibility.** Each lab's `run.sh` + manifests reproduce its artifacts on the same cluster.
5. **Honest gaps.** If a capability (e.g., full GPUDirect-TCPX) cannot be enabled in this environment, the guide says so explicitly and documents the standard-path behavior it did measure, plus what would change with the feature enabled.

---

## 7. Delivery

- New **public** GitHub repo `elideng-sg/hypercomputer-internode-deepdive`.
- Incremental, clean commits per module (doc + lab together where possible).
- Fresh repo (no branch protection) → push to `main`; module commits stay reviewable.
- Cluster mutations (deploying plugins, submitting jobs) are additive and reversible; teardown/cleanup steps are documented and capacity holders are preserved so GPU holds are never dropped.

---

## 8. Explicitly out of scope (YAGNI)

- **TPU communication (ICI/OCS)** — this guide is GPU/NCCL focused; TPUs are mentioned only for architectural contrast in Part I.
- **Model/accuracy quality of the workloads** — jobs and microbenchmarks exist to generate compute, communication, and execution behavior, not to train a useful model or reach a target accuracy.
- **Non-NVIDIA accelerators** and **Windows GPU stacks** — the labs target the Linux/COS + NVIDIA H100 environment of this cluster.

> Note on relationship to existing repos: `gcp-ai-infra-study` and `hypercomputer-training-jobs` remain useful companions, but this guide is now **fully self-contained** — GPU microarchitecture and driver/CUDA troubleshooting are taught here in depth (Part I), not delegated by reference.

---

## 9. Success criteria

- A reader can start with no GPU-cluster background and, following the guide top-to-bottom on an equivalent A3 cluster, go from understanding a single H100 to running and debugging a 16-GPU distributed training job.
- The guide explains **why** each number is what it is (mechanism → measurement) at every layer, not just how to run a command.
- Real captured data (arch specs, single-GPU roofline, NVLink bandwidth, inter-node bandwidth curves, topology, traces, failure signatures) is present for every lab that ran.
- Troubleshooting/debugging content maps concrete symptoms → root causes → fixes, backed by at least one deliberately injected real failure at both the single-GPU (driver/XID) and cluster (collective) levels.
