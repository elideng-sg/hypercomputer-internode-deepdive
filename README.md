# hypercomputer-internode-deepdive

A **standalone, hands-on guide** to the **NVIDIA GPU + AI-infrastructure stack** — how a GPU is built, how it executes work, how GPUs communicate within and across nodes, and how distributed workloads are scheduled, run, **monitored, profiled, benchmarked, and debugged** — taught with the **standard NVIDIA toolchain**.

One continuous, bottom-up journey:

**single node → inter-node communication → clustering & distributed execution**

…with a **NVIDIA tooling layer** cutting across all of it (nvidia-smi, DCGM, Nsight Systems/Compute, nvbandwidth, nccl-tests, perftest, dcgm-exporter + Grafana, and more).

Every mechanism and tool is paired with a **lab run live on a real 2-node A3 H100 cluster** (16 × H100 80GB), capturing **actual measured data** — architecture specs, roofline + profiler timelines, NVLink & inter-node bandwidth curves, topology dumps, NCCL traces, fleet dashboards, and failure signatures — not generic expected values.

> **Not limited to GCP.** The lab runs on Google Cloud A3 H100, but the mechanisms and tools are **platform-agnostic** — they transfer to on-prem/DGX, other clouds, bare metal, Slurm or Kubernetes. GCP-specific steps (DWS, GKE device plugin) are **flagged** with their generic equivalents (NVIDIA GPU Operator, `k8s-device-plugin`, Slurm `gres`).

> **Status: 🚧 planning.** The design spec is committed and under review. Implementation (docs + live labs) has not started yet.

---

## The NVIDIA tooling layer (cross-cutting)

Taught in the layer where each tool naturally appears, and consolidated in `docs/toolkit/`:

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
| 1 | GPU microarchitecture (SM, SIMT, memory hierarchy, Tensor Cores, Hopper) | nvidia-smi, NVML, deviceQuery | Inspect SMs/clocks/Tensor Cores + full `nvidia-smi -q` on a real H100 |
| 2 | Drivers, CUDA & troubleshooting (branches, install, XID, thermal) | DCGM diag, XID decode, nvidia-bug-report, gpu-burn | Verify driver/CUDA, run DCGM diagnostics, stress + read XID/throttle |
| 3 | Single-GPU execution & profiling (CUDA model, occupancy, roofline) | Nsight Systems/Compute, nvbandwidth | GEMM/FP8 + nvbandwidth; profile a kernel with nsys/ncu → roofline |
| 4 | Intra-node: NVLink / NVSwitch | nvidia-smi topo, nvbandwidth P2P, nccl-tests | Topology matrix; P2P + single-node 8-GPU all-reduce bandwidth |

### Part II — Inter-node Communication
| # | Layer | Key tools | Lab (run live) |
| :-- | :--- | :--- | :--- |
| 5 | NICs, RDMA, GPUDirect (gVNIC, TCPX/TCPXO/RoCE, plugin model) | perftest, ethtool/RoCE counters, ibstat | Inspect NICs & path; measure raw fabric BW; enable/compare Fast Socket/TCPX |
| 6 | NCCL collectives (ring vs tree, bus vs algo bandwidth) | nccl-tests, NCCL_DEBUG, topo export | 2-node all-reduce/all-gather sweeps → plotted GB/s curves |

### Part III — Clustering & Distributed Execution
| # | Layer | Key tools | Lab (run live) |
| :-- | :--- | :--- | :--- |
| 7 | GKE scheduling & topology (device plugin, gang, DWS; generic: GPU Operator/Slurm) | kubectl GPU inspection, DCGM DaemonSet | Schedule a 16-GPU gang across both nodes |
| 8 | Job frameworks (JobSet / Kueue, rendezvous) | JobSet/Kueue, kubectl | Multi-node JobSet across 16 GPUs |
| 9 | Distributed training & profiling (DDP / FSDP) | PyTorch profiler, nsys multi-rank, HTA | 2-node PyTorch job; DDP vs FSDP; multi-rank traces analyzed |
| 10 | Observability & debugging (fleet) | dcgm-exporter + Prometheus/Grafana, NCCL_DEBUG | Stand up dashboards; inject faults; read the signatures |

## The lab environment

| | |
| :--- | :--- |
| Cluster | `hypercomputer-a3-cluster` (GKE v1.33), `us-central1` |
| GPU pool | 2 × `a3-highgpu-8g` = **16 × H100 80GB** (DWS) |
| Journey | Single GPU → single node (8×NVLink) → 2 nodes → 16-GPU distributed jobs |

## Design spec

The full design is in
[`docs/superpowers/specs/2026-07-21-hypercomputer-internode-deepdive-design.md`](docs/superpowers/specs/2026-07-21-hypercomputer-internode-deepdive-design.md).

## Repository layout (planned)

```
docs/
  00-guide-overview.md            Journey, cluster, portability (GCP lab vs generic)
  toolkit/                        Cross-cutting NVIDIA tooling deep-dives (T1–T6)
  part1-single-node/              GPU arch, drivers/CUDA, single-GPU exec+profiling, NVLink
  part2-inter-node/               NICs/RDMA/GPUDirect, NCCL collectives
  part3-clustering-execution/     GKE scheduling, JobSet/Kueue, DDP/FSDP+profiling, fleet observability
labs/        Step-by-step practice; one dir per lab, with real captured output
manifests/   Reusable, verified Kubernetes YAML (incl. dcgm-exporter)
scripts/     Runner + capture/parse scripts
assets/      Captured real outputs: logs, CSVs, plots, profiler timelines, diagrams
reference/   Env-var cheat sheets, XID table, NCCL tunables, driver matrix, tool cheat-sheets, glossary
VERIFICATION.md   Provenance log of every live run and artifact
```
