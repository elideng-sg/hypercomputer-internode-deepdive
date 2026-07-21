# hypercomputer-internode-deepdive

A **standalone, hands-on guide** to the **NVIDIA GPU + AI-infrastructure stack** — from a single GPU's silicon, up through node and cluster communication, to the **purpose-built AI-factory platforms** NVIDIA ships (DGX/HGX, BlueField, Spectrum-X, DGX SuperPOD) — taught with the **standard NVIDIA toolchain** and backed by **labs run live on a real 2-node A3 H100 cluster** (16 × H100 80GB).

One continuous, bottom-up journey:

**single node → inter-node communication → clustering & distributed execution → platform & reference architectures**

…with a **NVIDIA tooling layer** cutting across all of it (nvidia-smi, DCGM, Nsight Systems/Compute, nvbandwidth, nccl-tests, perftest, dcgm-exporter + Grafana, and more).

Every mechanism and tool is paired with a lab capturing **actual measured data** — architecture specs, roofline + profiler timelines, NVLink & inter-node bandwidth curves, topology dumps, NCCL traces, fleet dashboards, and failure signatures — not generic expected values.

> **Scope: the full GCP AI Hypercomputer GPU portfolio — not just A3.** The lab runs live on A3 High (H100), but the guide covers and **attributes each concept to the right GCP product/family** — A3 High (TCPX), A3 Mega (TCPXO), A3 Ultra (H200, RoCE/GPUDirect-RDMA), A4 (B200), A4X (GB200 + NVLink domains) — and maps them to their NVIDIA-platform equivalents: GCP **Titanium** ↔ **BlueField DPU**, GPUDirect-TCPX/RDMA ↔ **Spectrum-X / InfiniBand**, GKE+JobSet+Kueue ↔ **Base Command Manager / Run:ai / Slurm**. Concepts and tools are otherwise **platform-agnostic** (on-prem/DGX/SuperPOD, other clouds, bare metal).

> **Honesty note on Part IV.** DGX/HGX *system software*, BlueField DPUs, and Spectrum-X are **not present** on the GCP A3 cluster (each A3 node rides an **HGX H100 baseboard**, but DGX OS / Fabric Manager / DPUs / Spectrum-X are not tenant-visible). Part IV is therefore **knowledge-first / reference-architecture**, with *observe-and-compare* exercises against what the A3 lab can actually show. It never claims that DGX/BlueField/Spectrum-X hardware was run here.

> **Status: 🚧 planning.** The design spec is committed and under review. Implementation (docs + live labs) has not started yet.

---

## The NVIDIA tooling layer (cross-cutting)

Taught in the layer where each tool appears, consolidated in `docs/toolkit/`:

| Category | Tools |
| :--- | :--- |
| Monitoring & inventory | `nvidia-smi` (+ `dmon`/`pmon`), NVML, **DCGM** (`dcgmi`), dcgm-exporter, nvtop, Prometheus/Grafana |
| Health & diagnostics | DCGM diagnostics (`dcgmi diag`), `nvidia-smi -q`, **XID** decode, `nvidia-bug-report.sh`, `gpu-burn`, ECC/RAS, throttle reasons |
| Profiling & tracing | **Nsight Systems** (`nsys`), **Nsight Compute** (`ncu`), CUPTI/NVTX, PyTorch profiler/Kineto, Holistic Trace Analysis |
| Benchmarking | `nvbandwidth`, `bandwidthTest`, **nccl-tests**, cuBLAS/cuDNN microbench, `gpu-burn`; methodology (warmup/repeats, bus vs algo bandwidth) |
| Networking / fabric | NCCL debug/topology, GPUDirect verification, `perftest` (`ib_write_bw`), `ethtool` + RoCE/ECN counters, `ibstat`/`mlxlink` |

## What this guide will cover

### Part I — Single Node
| # | Layer | Key tools | Lab (run live) |
| :-- | :--- | :--- | :--- |
| 1 | GPU microarchitecture (SM, SIMT, memory hierarchy, Tensor Cores, Hopper) | nvidia-smi, NVML, deviceQuery | Inspect SMs/clocks/Tensor Cores + full `nvidia-smi -q` |
| 2 | Drivers, CUDA & troubleshooting (branches, install, XID, thermal) | DCGM diag, XID decode, nvidia-bug-report, gpu-burn | Verify driver/CUDA, DCGM diag, stress + read XID/throttle |
| 3 | Single-GPU execution & profiling (CUDA model, occupancy, roofline) | Nsight Systems/Compute, nvbandwidth | GEMM/FP8 + nvbandwidth; profile a kernel → roofline |
| 4 | Intra-node: NVLink / NVSwitch — **the node as an HGX H100 baseboard** | nvidia-smi topo, nvbandwidth P2P, nccl-tests | Topology matrix mapped to HGX; P2P + 8-GPU all-reduce BW |

### Part II — Inter-node Communication
| # | Layer | Key tools | Lab (run live) |
| :-- | :--- | :--- | :--- |
| 5 | NICs, RDMA, GPUDirect (gVNIC, TCPX/TCPXO/RoCE, plugin model) | perftest, ethtool/RoCE counters, ibstat | Inspect NICs & path; raw fabric BW; enable/compare Fast Socket/TCPX |
| 6 | NCCL collectives (ring vs tree, bus vs algo bandwidth) | nccl-tests, NCCL_DEBUG, topo export | 2-node sweeps → plotted GB/s curves |

### Part III — Clustering & Distributed Execution
| # | Layer | Key tools | Lab (run live) |
| :-- | :--- | :--- | :--- |
| 7 | GKE scheduling & topology (device plugin, gang, DWS; generic: GPU Operator/Slurm) | kubectl GPU inspection, DCGM DaemonSet | Schedule a 16-GPU gang across both nodes |
| 8 | Job frameworks (JobSet / Kueue, rendezvous) | JobSet/Kueue, kubectl | Multi-node JobSet across 16 GPUs |
| 9 | Distributed training & profiling (DDP / FSDP) | PyTorch profiler, nsys multi-rank, HTA | 2-node PyTorch job; DDP vs FSDP; multi-rank traces analyzed |
| 10 | Observability & debugging (fleet) | dcgm-exporter + Prometheus/Grafana, NCCL_DEBUG | Stand up dashboards; inject faults; read the signatures |

### Part IV — Platform & Reference Architectures  *(knowledge-first; observe-and-compare)*
| # | Layer | Covers | Runnable here? |
| :-- | :--- | :--- | :--- |
| 11 | **DGX / HGX systems & troubleshooting** | HGX baseboard vs DGX system; DGX OS, NVSM, GPU Fabric Manager, Base Command Manager; system-level troubleshooting | Partial — map node to HGX; note DGX-only parts absent |
| 12 | **BlueField DPUs & DOCA** | DPU offload (net/storage/security), DOCA, BlueField-3; contrast w/ GCP Titanium | Read-only — links to NVIDIA DOCA skills |
| 13 | **Spectrum-X & AI fabrics** | Spectrum-X Ethernet vs Quantum InfiniBand (rail-optimized, SHARP) vs cloud fabric/TCPX | Read-only + compare to lab-06 curves |
| 14 | **DGX SuperPOD** | Scalable unit, compute/storage/mgmt fabrics, NVLink Switch System / GB200 NVL72, BCM/Run:ai, validation | Read-only — contrast with A3 scaling |

**Capstone `lab-11`:** map the A3 node to the HGX H100 baseboard, probe for (and document the absence of) Fabric Manager / BlueField / Spectrum-X, and build an "A3 tenant vs DGX SuperPOD" comparison from real output.

## The lab environment

| | |
| :--- | :--- |
| Cluster | `hypercomputer-a3-cluster` (GKE v1.33), `us-central1` |
| GPU pool | 2 × `a3-highgpu-8g` = **16 × H100 80GB** (each node = an HGX H100 baseboard) |
| Journey | Single GPU → single node (8×NVLink/HGX) → 2 nodes → 16-GPU jobs → platform reference architectures |

## Design spec

The full design is in
[`docs/superpowers/specs/2026-07-21-hypercomputer-internode-deepdive-design.md`](docs/superpowers/specs/2026-07-21-hypercomputer-internode-deepdive-design.md).

## Repository layout (planned)

```
docs/
  00-guide-overview.md            Journey, cluster, portability (GCP lab vs generic)
  toolkit/                        Cross-cutting NVIDIA tooling deep-dives (T1–T6)
  part1-single-node/              GPU arch, drivers/CUDA, single-GPU exec+profiling, NVLink/HGX
  part2-inter-node/               NICs/RDMA/GPUDirect, NCCL collectives
  part3-clustering-execution/     GKE scheduling, JobSet/Kueue, DDP/FSDP+profiling, fleet observability
  part4-platform-reference-arch/  DGX/HGX, BlueField/DOCA, Spectrum-X, DGX SuperPOD
labs/        Step-by-step practice; one dir per lab, with real captured output
manifests/   Reusable, verified Kubernetes YAML (incl. dcgm-exporter)
scripts/     Runner + capture/parse scripts
assets/      Captured real outputs: logs, CSVs, plots, profiler timelines, diagrams
reference/   Cheat sheets: XID table, NCCL tunables, driver matrix, tools, reference-arch, glossary
VERIFICATION.md   Provenance log of every live run and artifact
```
