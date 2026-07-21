# Design Spec: `hypercomputer-internode-deepdive`

**Date:** 2026-07-21 (rev 4)
**Author:** elideng-sg (with Claude Code)
**Status:** Draft — awaiting user review

---

## 1. Purpose

A **standalone, comprehensive, hands-on guide** to the **NVIDIA GPU + AI-infrastructure stack** — from a single GPU's silicon, up through node and cluster communication, to the **purpose-built AI-factory platforms** NVIDIA ships (DGX/HGX systems, BlueField DPUs, Spectrum-X fabrics, DGX SuperPOD) — teaching both the **mechanism** (why it works) and the **practice** (how to run, monitor, profile, benchmark, and debug it) with the **standard NVIDIA toolchain**, backed by **labs executed live on a real 2-node A3 H100 cluster**.

The guide walks one continuous, bottom-up journey:

1. **Single node** — GPU microarchitecture, driver/CUDA stack + troubleshooting, single-GPU execution & profiling, and the 8-GPU NVLink/NVSwitch (HGX) fabric.
2. **Inter-node communication** — NICs, RDMA, GPUDirect, and NCCL collectives.
3. **Clustering & distributed execution** — GKE scheduling, JobSet/Kueue gang scheduling, `torchrun` rendezvous, DDP/FSDP, and fleet-scale observability.
4. **Platform & reference architectures** — how NVIDIA builds AI factories: DGX/HGX systems and system-level troubleshooting, BlueField DPUs & DOCA, Spectrum-X vs. InfiniBand vs. cloud fabrics, and the DGX SuperPOD reference architecture — each **contrasted with what the GCP A3 lab demonstrates**.

Cutting **across all parts** is a **NVIDIA tooling layer** — monitoring, health/diagnostic, profiling, benchmarking, and fabric tools (nvidia-smi, DCGM, Nsight Systems/Compute, nvbandwidth, nccl-tests, perftest, dcgm-exporter, and more) — taught in context and consolidated in a cross-cutting reference.

### Platform scope — not limited to GCP
The **concrete lab** runs on a Google Cloud A3 H100 GKE cluster, but the **mechanisms and tools are platform-agnostic** and transfer to any NVIDIA GPU environment — on-prem / DGX / SuperPOD, other clouds, bare metal, Slurm or Kubernetes. Where a step is GCP-specific (e.g. DWS provisioning, GKE device-plugin DaemonSets), it is **explicitly flagged** and the **generic equivalent** noted (NVIDIA GPU Operator, `k8s-device-plugin`, Slurm `gres`, Base Command Manager). Part IV makes the cloud-vs-purpose-built-platform contrast explicit.

### Reality check on Part IV (honesty principle)
DGX/HGX **system software**, **BlueField DPUs**, and **Spectrum-X** switches are **not present** on the GCP A3 cluster (GCP fronts its own host-offload/"Titanium" and network fabric; A3 nodes ride an **HGX H100 baseboard** but without DGX OS / NVSM / Fabric Manager exposed to the tenant). Part IV is therefore **knowledge-first / reference-architecture**, with **observe-and-compare** exercises limited to what the A3 lab can actually reveal (e.g. mapping `nvidia-smi topo` to the HGX baseboard, detecting the presence/absence of DPUs, SuperNICs, and a tenant-visible Fabric Manager). Each Part IV section states clearly **what is runnable here vs. read-only**, and for hands-on BlueField/DOCA work **points to the dedicated NVIDIA DOCA skills** rather than duplicating them.

### Target reader
An engineer or data scientist who can use `kubectl` and Python and wants **deep, first-principles knowledge** of how GPU workloads are built, communicate, execute, and are operated — and how the underlying NVIDIA platforms (DGX/HGX/SuperPOD, BlueField, Spectrum-X) are architected — enough to **design, implement, troubleshoot, benchmark, and debug** them from a single card to a SuperPOD-class fabric, on GCP or elsewhere. Depth target matches the existing `gcp-ai-infra-study` notes (textbook rigor + practitioner detail).

---

## 2. Live environment (the lab)

| Property | Value |
| :--- | :--- |
| Project | `hdlab-elideng` |
| Cluster | `hypercomputer-a3-cluster` (GKE `v1.33`) |
| Region | `us-central1` |
| GPU pool | `a3-h100-dws-pool` — **2 × `a3-highgpu-8g`** = **16 × H100 80GB** (each node = an HGX H100 8-GPU baseboard) |
| Provisioning | Dynamic Workload Scheduler (DWS) queued provisioning, 7-day holds |
| Aux pool | `default-pool` (e2-standard-4) for control/CPU workloads |

**Verified cluster facts to characterize during execution (not assumed):**
- Nodes currently expose only `nvidia.com/gpu: 8` as an extended resource — **no dedicated GPU-NIC / GPUDirect-TCPX extended resources are present**.
- `nccl-fastsocket-installer` DaemonSet exists but has **0 pods scheduled**; **no `tcpx`/`tcpxo` DaemonSet** is installed.
- **No tenant-visible BlueField DPU, Spectrum-X SuperNIC, or DGX Fabric Manager** is expected — to be confirmed and documented as the platform contrast in Part IV.
- **Implication:** as provisioned, inter-node NCCL traffic most likely traverses the standard gVNIC/VPC TCP path rather than GPUDirect. **Characterizing this actual data path — and, where feasible, enabling GPUDirect-TCPX / NCCL Fast Socket and benchmarking the before/after delta — is a core teaching thread**, not an assumption to paper over.

A machine-generated `VERIFICATION.md` records exactly what was run, when, on which nodes, and which artifact resulted, so every documented number is traceable and reproducible.

---

## 3. Repository architecture

```
hypercomputer-internode-deepdive/
├── README.md                 # Guide map, cluster-at-a-glance, reading order, portability, how to run labs
├── docs/
│   ├── 00-guide-overview.md              # What/why, the journey, the cluster, PORTABILITY (GCP lab vs generic)
│   ├── toolkit/                          # Cross-cutting NVIDIA tooling deep-dives (used by every layer)
│   │   ├── T1-monitoring-inventory.md
│   │   ├── T2-health-diagnostics.md
│   │   ├── T3-profiling-tracing.md
│   │   ├── T4-benchmarking.md
│   │   ├── T5-networking-fabric-tools.md
│   │   └── T6-portability-matrix.md
│   ├── part1-single-node/
│   │   ├── 01-gpu-microarchitecture.md
│   │   ├── 02-drivers-cuda-install-troubleshooting.md
│   │   ├── 03-single-gpu-execution-and-profiling.md
│   │   └── 04-intranode-nvlink-nvswitch-hgx.md    # NVLink/NVSwitch framed on the HGX H100 baseboard
│   ├── part2-inter-node/
│   │   ├── 05-nic-rdma-gpudirect.md
│   │   └── 06-nccl-collectives.md
│   ├── part3-clustering-execution/
│   │   ├── 07-gke-scheduling-topology.md
│   │   ├── 08-job-frameworks-jobset-kueue.md
│   │   ├── 09-distributed-training-ddp-fsdp.md
│   │   └── 10-observability-debugging.md
│   └── part4-platform-reference-arch/    # Knowledge-first; observe-and-compare vs the A3 lab
│       ├── 11-dgx-hgx-systems.md         # HGX baseboard vs DGX system; DGX OS, NVSM, Fabric Manager, BCM; system-level troubleshooting
│       ├── 12-bluefield-dpu-doca.md      # DPU offload model, DOCA, BlueField-3; contrast w/ GCP Titanium; links to NVIDIA DOCA skills
│       ├── 13-spectrum-x-and-fabrics.md  # Spectrum-X Ethernet vs Quantum InfiniBand (rail-optimized, SHARP) vs cloud fabric/TCPX
│       └── 14-dgx-superpod.md            # SuperPOD reference arch: scalable unit, fabrics, NVLink Switch System / GB200 NVL72, BCM/Run:ai, validation
├── labs/                     # Step-by-step practice; one dir per lab, with real captured output
│   ├── lab-01-gpu-arch-inspect/
│   ├── lab-02-driver-cuda-health/
│   ├── lab-03-single-gpu-benchmark-profile/
│   ├── lab-04-intranode-nvlink-hgx/
│   ├── lab-05-network-path-inspect/
│   ├── lab-06-nccl-tests-internode/
│   ├── lab-07-gke-gang-schedule/
│   ├── lab-08-jobset-multinode/
│   ├── lab-09-2node-pytorch-ddp-fsdp-profile/
│   ├── lab-10-observability-fleet-debug/
│   └── lab-11-platform-compare/          # Observe-and-compare: map node to HGX; detect DPU/SuperNIC/Fabric-Manager presence; A3-vs-DGX/SuperPOD contrast
├── manifests/                # Reusable, verified K8s YAML (JobSet, PyTorchJob, DaemonSets, dcgm-exporter, ConfigMaps)
├── scripts/                  # Runner + capture/parse scripts (collect nvidia-smi/DCGM/NCCL/nsys/topology → CSV/plots)
├── assets/                   # Captured REAL outputs: logs, CSVs, rendered plots, topology diagrams, profiler timelines
├── reference/                # Env-var cheat sheets, XID table, NCCL tunables, driver matrix, tool cheat-sheets, reference-arch cheat-sheet, glossary
└── VERIFICATION.md           # Provenance log of every live run and artifact
```

**Layout contract for each `labs/lab-XX/`:** `README.md` (objective, prerequisites, numbered steps, **real captured output**, interpretation), the manifest(s)/`run.sh` it uses, and links back to its `docs/` layer + relevant `docs/toolkit/` docs. Part IV labs additionally state **runnable-here vs read-only** at the top.

**Doc ↔ lab ↔ toolkit pairing:** each layer doc ends with "Practice" (→ lab) and "Tools in this layer" (→ toolkit) sections; each lab links back. Three-way spine.

---

## 4. Content spine (single node → inter-node → cluster → platform, tools woven throughout)

### Part I — Single Node

| # | Layer | Mechanism doc covers | Tools | Lab (live) |
| :-- | :--- | :--- | :--- | :--- |
| 1 | **GPU microarchitecture** | CPU/GPU/TPU; SM/SIMT/warps; memory hierarchy; Tensor Cores & FP8/BF16; H100 (Hopper) — HBM3, TMA, thread-block clusters | nvidia-smi (`-q`, topo), NVML, deviceQuery | `lab-01`: inspect SMs/clocks/Tensor Cores + full `nvidia-smi -q` |
| 2 | **Drivers, CUDA & troubleshooting** | Driver branches & matrix; GKE managed vs GCE manual install (+ generic GPU Operator); CUDA vs driver; XID taxonomy; thermal/throttle | DCGM `dcgmi diag`, `nvidia-smi -q -d`, XID decode, `nvidia-bug-report.sh`, `gpu-burn` | `lab-02`: verify driver/CUDA, DCGM diag, gpu-burn stress, read XID/throttle |
| 3 | **Single-GPU execution & profiling** | CUDA execution model (streams, kernels, occupancy, roofline); step anatomy; compute- vs memory-bound | Nsight Systems (`nsys`), Nsight Compute (`ncu`), NVTX, `nvbandwidth` | `lab-03`: GEMM/FP8 + `nvbandwidth`; `nsys`/`ncu` profile → roofline |
| 4 | **Intra-node: NVLink / NVSwitch (HGX)** | 8-GPU HGX H100 fabric: gen4 NVLink, NVSwitch, ~900 GB/s bisection; NVLink vs PCIe; why intra ≫ inter; **the node as an HGX baseboard** | `nvidia-smi topo -m`, `nvbandwidth` P2P, `nccl-tests` (single node) | `lab-04`: topology matrix mapped to HGX; P2P + single-node 8-GPU all-reduce BW |

### Part II — Inter-node Communication

| # | Layer | Mechanism doc covers | Tools | Lab (live) |
| :-- | :--- | :--- | :--- | :--- |
| 5 | **NICs, RDMA, GPUDirect** | gVNIC vs GPU NICs; RDMA/RoCE vs GPUDirect-TCPX vs TCPXO; NCCL network-plugin model; `NCCL_*`; characterizing the real path | `perftest` (`ib_write_bw`), `ethtool`+RoCE/ECN counters, `ibstat`/`mlxlink`, NCCL topo | `lab-05`: inspect NICs & path; raw fabric BW; enable/compare Fast Socket/TCPX |
| 6 | **NCCL collectives** | all-reduce/all-gather/reduce-scatter; ring vs tree; topology discovery; size-vs-BW; bus vs algo BW | `nccl-tests`, `NCCL_DEBUG`/`_SUBSYS`, topo export | `lab-06`: 2-node sweep, ring vs tree, standard vs plugin → plotted curves |

### Part III — Clustering & Distributed Execution

| # | Layer | Mechanism doc covers | Tools | Lab (live) |
| :-- | :--- | :--- | :--- | :--- |
| 7 | **GKE scheduling & topology** | Device plugin; requests/tolerations; topology-aware & gang; DWS + holders (generic: GPU Operator, NFD, Slurm gres) | kubectl GPU inspection, DCGM DaemonSet | `lab-07`: schedule a 16-GPU gang |
| 8 | **Job frameworks** | JobSet + Kueue; headless service & pod DNS rendezvous; gang scheduling | JobSet/Kueue, kubectl | `lab-08`: multi-node JobSet across 16 GPUs |
| 9 | **Distributed training & profiling** | DDP vs FSDP; bucketing → collectives; `torchrun` rendezvous; step → NCCL; scaling efficiency | PyTorch profiler/Kineto, `nsys` multi-rank, Holistic Trace Analysis | `lab-09`: 2-node job; DDP vs FSDP; multi-rank traces → HTA |
| 10 | **Observability & debugging (fleet)** | Metrics pipeline; slow/stalled/mismatched collective diagnosis; XID at scale; hangs/timeouts | dcgm-exporter + Prometheus/Grafana, `NCCL_DEBUG`, `ethtool` | `lab-10`: dcgm-exporter→Grafana; inject faults; read signatures |

### Part IV — Platform & Reference Architectures (knowledge-first; observe-and-compare)

| # | Layer | Mechanism doc covers | Runnable here? |
| :-- | :--- | :--- | :--- |
| 11 | **DGX / HGX systems & troubleshooting** | HGX baseboard vs full DGX H100/H200 system; DGX OS, NVSM, **GPU Fabric Manager**, Base Command Manager; system health & NVLink/NVSwitch/BMC-sideband troubleshooting; how a tenant A3 node relates to a DGX | Partial — map node to HGX; note DGX-only components absent |
| 12 | **BlueField DPUs & DOCA** | DPU offload model (net/storage/security), DOCA stack, BlueField-3 SuperNIC; contrast with GCP Titanium/IPU offload; where DPUs sit in an AI cluster | Read-only — no DPU on A3; **links to NVIDIA DOCA skills** for hands-on |
| 13 | **Spectrum-X & AI fabrics** | Spectrum-X Ethernet (Spectrum-4 + BlueField-3 SuperNIC, adaptive routing, RoCE congestion control) vs **Quantum InfiniBand** (rail-optimized, **SHARP** in-network reduction) vs cloud fabric/GPUDirect-TCPX; mapped to what NCCL sees | Read-only + compare to lab-06 curves |
| 14 | **DGX SuperPOD** | Reference architecture: scalable unit (SU); compute/storage/management fabrics; rail-optimized topology; **NVLink Switch System / GB200 NVL72** domains; Base Command Manager / Run:ai; validation & acceptance; scaling vs cloud A3 | Read-only — contrast with how A3 scales |

**Capstone lab (`lab-11`):** observe-and-compare — map the A3 node to the HGX H100 baseboard diagram, probe for (and document the absence of) DGX Fabric Manager / BlueField DPU / Spectrum-X SuperNIC, and produce a concrete "A3 tenant vs DGX SuperPOD" comparison table backed by real `nvidia-smi`/`kubectl`/topology output.

Single-GPU/node microbenchmarks (Part I) establish the **baseline** that makes inter-node/cluster/platform comparisons meaningful. Sweeps run **thoroughly** (repeated runs, multiple algorithms), keeping the GPUs busy.

---

## 5. Live-lab data flow

```
run.sh → submit workload (nvidia-smi / CUDA microbench / nsys|ncu / kubectl / JobSet / PyTorchJob / platform probe)
       → scripts/ capture nvidia-smi + DCGM + NCCL logs + profiler traces + topology + timings
       → parse to CSV (scripts/parse_*.py) → render plots/tables/timelines → assets/
       → lab README embeds ACTUAL numbers + interpretation
       → VERIFICATION.md appends provenance (what/when/where/artifact)
```

Re-running a lab regenerates its artifacts deterministically. Long-running sweeps keep the GPUs busy, consistent with the standing "never leave the GPU idle" practice.

---

## 6. Correctness & verification principles

1. **Execute before documenting.** Every command/manifest/number is run live first; captured output embedded, not invented.
2. **Verify hardware claims against running nodes.** Networking family, NIC count/type, NVLink gen, SM/clock/Tensor-Core specs, actual NCCL path, and platform component presence/absence confirmed on the nodes.
3. **Traceable provenance** via `VERIFICATION.md`.
4. **Reproducibility** — each lab's `run.sh` + manifests reproduce its artifacts.
5. **Honest gaps & portability.** Where a capability or platform component is absent (GPUDirect-TCPX, DGX Fabric Manager, BlueField, Spectrum-X), the guide says so, documents what it did measure, and describes what changes on a DGX/SuperPOD or other platform. **Part IV is explicitly labeled knowledge-first**, never implying we ran DGX/BlueField/Spectrum-X hardware.

---

## 7. Delivery

- Public GitHub repo `elideng-sg/hypercomputer-internode-deepdive`.
- Incremental, clean commits per module (doc + lab together where possible); push to `main`.
- Cluster mutations are additive/reversible; teardown documented; **capacity holders preserved so GPU holds are never dropped**.
- No NVIDIA skills installed without explicit user approval; Part IV **references** the DOCA/BlueField skills for hands-on rather than duplicating them.

---

## 8. Explicitly out of scope (YAGNI)

- **Hands-on DGX/HGX system administration, BlueField/DOCA programming, and Spectrum-X switch configuration** — taught conceptually in Part IV and routed to NVIDIA's dedicated DOCA skills / docs; not executed here (hardware absent).
- **TPU communication (ICI/OCS)** — mentioned only for architectural contrast.
- **Model/accuracy quality** — workloads generate compute/comm/execution behavior, not target accuracy.
- **Windows GPU stacks.**
- **Exhaustive per-product SDK internals** — the guide teaches the day-to-day infra engineer's stack, not every NVIDIA SDK.

> The guide is **fully self-contained** and **platform-agnostic in its concepts**; GCP-specific steps are flagged with generic equivalents, and purpose-built-platform material (Part IV) is contrasted against the cloud A3 lab.

---

## 9. Success criteria

- A reader can go from understanding a single H100 to running, profiling, benchmarking, and debugging a 16-GPU distributed job on an equivalent NVIDIA cluster — **and** understand how that maps up to DGX/HGX systems, BlueField/Spectrum-X networking, and a DGX SuperPOD.
- The guide explains **why** each number is what it is (mechanism → measurement) at every layer.
- For every major NVIDIA tool covered, there is at least one lab where it is **actually run** on the live cluster with real output shown and interpreted.
- Real captured data is present for every runnable lab; Part IV clearly separates **measured** facts (from the A3 lab) from **reference-architecture** knowledge.
- Troubleshooting content maps symptoms → root causes → fixes, backed by ≥1 deliberately injected real failure at the single-GPU (driver/XID) and cluster (collective) levels, plus a **system-level troubleshooting walkthrough** (NVSM/Fabric-Manager-style) in Part IV described against the DGX platform.
- GCP-specific steps are flagged with generic equivalents; Part IV never claims DGX/BlueField/Spectrum-X hardware was run.
