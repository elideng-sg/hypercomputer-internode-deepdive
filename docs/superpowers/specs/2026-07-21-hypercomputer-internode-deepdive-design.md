# Design Spec: `hypercomputer-internode-deepdive`

**Date:** 2026-07-21 (rev 3)
**Author:** elideng-sg (with Claude Code)
**Status:** Draft — awaiting user review

---

## 1. Purpose

A **standalone, comprehensive, hands-on guide** to the **NVIDIA GPU + AI-infrastructure stack** — how a GPU is built, how it executes work, how GPUs communicate within and across nodes, and how distributed workloads are scheduled, run, monitored, profiled, benchmarked, and debugged — taught with the **standard NVIDIA toolchain** and backed by **labs executed live on a real 2-node A3 H100 cluster**.

The guide walks one continuous, bottom-up journey:

1. **Single node** — GPU microarchitecture, driver/CUDA stack + troubleshooting, single-GPU execution & profiling, and the 8-GPU NVLink/NVSwitch fabric.
2. **Inter-node communication** — NICs, RDMA, GPUDirect, and NCCL collectives.
3. **Clustering & distributed execution** — GKE scheduling, JobSet/Kueue gang scheduling, `torchrun` rendezvous, DDP/FSDP, and fleet-scale observability.

Cutting **across all three parts** is a **NVIDIA tooling layer** — the monitoring, health/diagnostic, profiling, benchmarking, and fabric tools an engineer actually uses (nvidia-smi, DCGM, Nsight Systems/Compute, nvbandwidth, nccl-tests, perftest, and more). Tools are taught **in the layer where they naturally appear** and consolidated in a cross-cutting reference.

### Platform scope — not limited to GCP
The **concrete lab** runs on a Google Cloud A3 H100 GKE cluster, but the **mechanisms and tools are platform-agnostic** and transfer to any NVIDIA GPU environment — on-prem / DGX, other clouds, bare metal, Slurm or Kubernetes. Where a step is GCP-specific (e.g. DWS provisioning, GKE device-plugin DaemonSets), it is **explicitly flagged** and the **generic equivalent** is noted (e.g. NVIDIA GPU Operator, `k8s-device-plugin`, Slurm `gres`). A short **portability note** in each such section keeps the guide useful off-GCP.

### Target reader
An engineer or data scientist who can use `kubectl` and Python and wants **deep, first-principles knowledge** of how GPU workloads are built, communicate, execute, and are operated on modern AI infrastructure — enough to **design, implement, troubleshoot, benchmark, and debug** them from a single card to a cluster, on GCP or elsewhere. Depth target matches the existing `gcp-ai-infra-study` notes (textbook rigor + practitioner detail).

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
├── README.md                 # Guide map, cluster-at-a-glance, reading order, portability, how to run labs
├── docs/
│   ├── 00-guide-overview.md              # What/why, the journey, the cluster, PORTABILITY (GCP lab vs generic)
│   ├── toolkit/                          # Cross-cutting NVIDIA tooling deep-dives (used by every layer)
│   │   ├── T1-monitoring-inventory.md    # nvidia-smi, dmon/pmon, NVML, DCGM/dcgmi, dcgm-exporter, nvtop, Prometheus/Grafana
│   │   ├── T2-health-diagnostics.md      # DCGM diag levels, nvidia-smi -q, XID taxonomy + decode, nvidia-bug-report, gpu-burn, ECC/RAS, throttle reasons
│   │   ├── T3-profiling-tracing.md       # Nsight Systems (nsys), Nsight Compute (ncu), CUPTI, NVTX, PyTorch profiler/Kineto, Holistic Trace Analysis
│   │   ├── T4-benchmarking.md            # nvbandwidth, bandwidthTest, nccl-tests, cuBLAS/cuDNN microbench, gpu-burn, methodology (warmup/repeats, bus vs algo BW)
│   │   ├── T5-networking-fabric-tools.md # NCCL debug/topo, GPUDirect verification, perftest (ib_write_bw/ib_send_bw), ethtool + RoCE/ECN counters, mlxlink/ibstat
│   │   └── T6-portability-matrix.md      # Tool/mechanism → bare-metal / DGX / other clouds / Slurm / K8s; GCP-specific bits flagged with generic equivalents
│   ├── part1-single-node/
│   │   ├── 01-gpu-microarchitecture.md
│   │   ├── 02-drivers-cuda-install-troubleshooting.md
│   │   ├── 03-single-gpu-execution-and-profiling.md
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
│   ├── lab-02-driver-cuda-health/
│   ├── lab-03-single-gpu-benchmark-profile/
│   ├── lab-04-intranode-nvlink/
│   ├── lab-05-network-path-inspect/
│   ├── lab-06-nccl-tests-internode/
│   ├── lab-07-gke-gang-schedule/
│   ├── lab-08-jobset-multinode/
│   ├── lab-09-2node-pytorch-ddp-fsdp-profile/
│   └── lab-10-observability-fleet-debug/
├── manifests/                # Reusable, verified K8s YAML (JobSet, PyTorchJob, DaemonSets, dcgm-exporter, ConfigMaps)
├── scripts/                  # Runner + capture/parse scripts (collect nvidia-smi/DCGM/NCCL/nsys/topology → CSV/plots)
├── assets/                   # Captured REAL outputs: logs, CSVs, rendered plots, topology diagrams, profiler timelines
├── reference/                # Env-var cheat sheets, XID table, NCCL tunables, driver matrix, tool cheat-sheets, glossary
└── VERIFICATION.md           # Provenance log of every live run and artifact
```

**Layout contract for each `labs/lab-XX/`:**
- `README.md` — objective, prerequisites, numbered steps, **real captured output**, and interpretation ("what this number means / what good vs bad looks like").
- The manifest(s) and `run.sh` it uses (or references into `manifests/`).
- Links back to the `docs/` layer whose mechanism it exercises, and to the `docs/toolkit/` docs for the tools it uses.

**Doc ↔ lab ↔ toolkit pairing:** each layer doc ends with a "Practice" section linking to its lab and a "Tools in this layer" subsection linking to the relevant `toolkit/` docs; each lab links back up. This is the three-way spine.

---

## 4. Content spine (single node → inter-node → cluster, tools woven throughout)

Each layer = **mechanism doc + matching lab**, with the NVIDIA tools used at that layer called out and detailed in `docs/toolkit/`.

### Part I — Single Node

| # | Layer | Mechanism doc covers | Tools introduced | Lab (live) |
| :-- | :--- | :--- | :--- | :--- |
| 1 | **GPU microarchitecture** | CPU vs GPU vs TPU; SM/SIMT/warps/divergence; memory hierarchy; Tensor Cores & FP8/BF16; H100 (Hopper) — HBM3, TMA, thread-block clusters | `nvidia-smi` (`-q`, topo), NVML, `deviceQuery` | `lab-01`: inspect SMs/clocks/Tensor Cores + full `nvidia-smi -q` on a real H100 |
| 2 | **Drivers, CUDA & troubleshooting** | Driver branches (LTSB/PB/NFB) & matrix; GKE managed install vs GCE manual (+ generic GPU Operator); CUDA toolkit/runtime vs driver; XID taxonomy; thermal/throttle; diagnostics workflow | **DCGM** `dcgmi diag`, `nvidia-smi -q -d …`, XID decode, `nvidia-bug-report.sh`, `gpu-burn`, ECC/RAS | `lab-02`: verify driver/CUDA, run DCGM diagnostics, stress with gpu-burn, read/interpret XID + throttle reasons |
| 3 | **Single-GPU execution & profiling** | CUDA execution model (streams, kernels, occupancy, roofline); how one training/inference step runs; compute- vs memory-bound analysis | **Nsight Systems** (`nsys`), **Nsight Compute** (`ncu`), NVTX, `nvbandwidth`, `bandwidthTest` | `lab-03`: single-GPU GEMM/FP8 + `nvbandwidth`; profile a kernel with `nsys`/`ncu`; place it on a roofline → measured TFLOPs & GB/s |
| 4 | **Intra-node: NVLink / NVSwitch** | 8-GPU fabric: gen4 NVLink, NVSwitch, ~900 GB/s bisection; NVLink vs PCIe; why intra-node ≫ inter-node | `nvidia-smi topo -m`, `nvbandwidth` (P2P), `nccl-tests` (single node) | `lab-04`: topology matrix; `nvbandwidth` P2P; single-node 8-GPU all-reduce → measured per-link/bus BW |

### Part II — Inter-node Communication

| # | Layer | Mechanism doc covers | Tools introduced | Lab (live) |
| :-- | :--- | :--- | :--- | :--- |
| 5 | **NICs, RDMA, GPUDirect** | gVNIC vs GPU NICs; RDMA/RoCE vs GPUDirect-TCPX vs TCPXO; NCCL network-plugin DaemonSet model; `NCCL_*`/`NCCL_GPUDIRECTTCPX_*`; characterizing the real path | `perftest` (`ib_write_bw`/`ib_send_bw`), `ethtool` + RoCE/ECN counters, `ibstat`/`mlxlink`, NCCL topo dump | `lab-05`: inspect NICs & plugin path; measure raw fabric BW; (if feasible) deploy Fast Socket / TCPX and prepare A/B |
| 6 | **NCCL collectives** | all-reduce/all-gather/reduce-scatter; ring vs tree; topology discovery; message-size vs BW; bus vs algo BW; how the path changes the curve | `nccl-tests` suite, `NCCL_DEBUG`/`NCCL_DEBUG_SUBSYS`, NCCL topology export | `lab-06`: 2-node all-reduce/all-gather **thorough size sweep**, ring vs tree, standard-path vs plugin → plotted GB/s curves |

### Part III — Clustering & Distributed Execution

| # | Layer | Mechanism doc covers | Tools introduced | Lab (live) |
| :-- | :--- | :--- | :--- | :--- |
| 7 | **GKE scheduling & topology** | Device plugin; requests/limits & tolerations; topology-aware & gang scheduling; DWS + capacity holders (generic: GPU Operator, NFD, Slurm gres) | `kubectl` GPU inspection, device-plugin/DCGM DaemonSet checks | `lab-07`: schedule a 16-GPU gang across both nodes |
| 8 | **Job frameworks** | JobSet (replicated jobs) + Kueue (queue/quota); headless service & pod DNS rendezvous; gang scheduling | JobSet/Kueue CLIs, `kubectl` job introspection | `lab-08`: multi-node JobSet across 16 GPUs |
| 9 | **Distributed training & profiling** | DDP vs FSDP; gradient bucketing → collectives; `torchrun`/`torchx` rendezvous; step → NCCL mapping; scaling efficiency | **PyTorch profiler**/Kineto, **Nsight Systems** multi-rank, **Holistic Trace Analysis**, `NCCL_DEBUG` | `lab-09`: 2-node/16-GPU PyTorch job; DDP vs FSDP; capture multi-rank `nsys` + PyTorch traces; analyze with HTA |
| 10 | **Observability & debugging (fleet)** | Fleet metrics pipeline; slow/stalled/mismatched collective diagnosis; XID at scale; hangs & timeouts | **dcgm-exporter** + Prometheus/Grafana, DCGM health checks, `NCCL_DEBUG`, `ethtool` counters | `lab-10`: stand up dcgm-exporter→Grafana; inject faults (kill rank, saturate NIC, version mismatch) and read the signatures |

Single-GPU and single-node microbenchmarks (Part I) establish the **baseline numbers** that make inter-node/cluster measurements meaningful. Message-size sweeps and ring/tree comparisons run **thoroughly** (repeated runs, multiple algorithms), making best use of the held GPUs.

---

## 5. Live-lab data flow

For each lab:

```
run.sh → submit workload (nvidia-smi / CUDA microbench / nsys|ncu / kubectl / JobSet / PyTorchJob)
       → scripts/ capture nvidia-smi + DCGM + NCCL logs + profiler traces + topology + timings
       → parse to CSV (scripts/parse_*.py)
       → render plots/tables/timelines → assets/
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
5. **Honest gaps & portability.** If a capability (e.g. full GPUDirect-TCPX, or a tool requiring privileges GKE COS doesn't grant) cannot be exercised here, the guide says so, documents what it did measure, and notes what changes on other platforms.

---

## 7. Delivery

- New **public** GitHub repo `elideng-sg/hypercomputer-internode-deepdive`.
- Incremental, clean commits per module (doc + lab together where possible).
- Fresh repo (no branch protection) → push to `main`; module commits stay reviewable.
- Cluster mutations (deploying plugins, exporters, submitting jobs) are additive and reversible; teardown/cleanup steps are documented and capacity holders are preserved so GPU holds are never dropped.

---

## 8. Explicitly out of scope (YAGNI)

- **TPU communication (ICI/OCS)** — this guide is GPU/NCCL/NVIDIA-tooling focused; TPUs are mentioned only for architectural contrast in Part I.
- **Model/accuracy quality of the workloads** — jobs and microbenchmarks exist to generate compute, communication, and execution behavior, not to reach a target accuracy.
- **Windows GPU stacks** — labs target the Linux/COS + NVIDIA H100 environment; Windows tooling differences are out of scope.
- **Exhaustive per-product tool internals** — the toolkit section teaches the tools an infra engineer uses day to day; it is not a manual for every NVIDIA SDK.

> Note: the guide is **fully self-contained** and **platform-agnostic in its concepts** — GPU microarchitecture, driver/CUDA troubleshooting, and the NVIDIA toolchain are taught here in depth (not delegated by reference), with GCP-specific steps flagged and generic equivalents given.

---

## 9. Success criteria

- A reader can start with no GPU-cluster background and, following the guide top-to-bottom on an equivalent NVIDIA cluster (GCP or otherwise), go from understanding a single H100 to running, profiling, benchmarking, and debugging a 16-GPU distributed training job.
- The guide explains **why** each number is what it is (mechanism → measurement) at every layer, not just how to run a command.
- For every major NVIDIA tool covered, there is at least one lab where it is **actually run** against the live cluster with real output shown and interpreted.
- Real captured data (arch specs, single-GPU roofline + profiler timelines, NVLink/P2P bandwidth, inter-node bandwidth curves, topology, NCCL traces, fleet dashboards, failure signatures) is present for every lab that ran.
- Troubleshooting content maps concrete symptoms → root causes → fixes, backed by at least one deliberately injected real failure at both the single-GPU (driver/XID) and cluster (collective) levels.
- GCP-specific steps are clearly flagged with their generic (non-GCP) equivalents, so the guide is usable on other NVIDIA platforms.
