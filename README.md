# hypercomputer-internode-deepdive

A **standalone, hands-on guide** that walks you from a **single GPU** to a **multi-node GPU cluster** on **Google Cloud AI Hypercomputer** — teaching both the **mechanism** (why it works) and the **practice** (how to run it) at every layer.

The journey is one continuous bottom-up story:

**single node → inter-node communication → clustering & distributed execution**

Every mechanism is paired with a **lab run live on a real 2-node A3 H100 cluster** (16 × H100 80GB), capturing **actual measured data** — architecture specs, roofline numbers, NVLink & inter-node bandwidth curves, topology dumps, NCCL traces, and failure signatures — not generic expected values.

The guide is **fully self-contained**: GPU microarchitecture and driver/CUDA troubleshooting are taught here in depth, so you can go from understanding one H100 to debugging a 16-GPU distributed job without leaving the repo.

> **Status: 🚧 planning.** The design spec is committed and under review. Implementation (docs + live labs) has not started yet.

---

## What this guide will cover

Each layer gets a mechanism deep-dive **and** a hands-on lab.

### Part I — Single Node
| # | Layer | Lab (run live) |
| :-- | :--- | :--- |
| 0 | Overview & the cluster | Enumerate the real cluster: nodes, GPUs, NICs, DaemonSets |
| 1 | GPU microarchitecture (SM, SIMT, memory hierarchy, Tensor Cores, Hopper) | `nvidia-smi` / `deviceQuery` — SMs, clocks, Tensor Core support on a real H100 |
| 2 | Drivers, CUDA & troubleshooting (branches, GKE install, XID, thermal) | Verify driver/CUDA, read `dmesg`/XID + DCGM diagnostics, interpret a fault |
| 3 | Single-GPU execution (CUDA model, streams, occupancy, roofline) | Single-GPU GEMM/FP8 + memory-bandwidth microbench → measured TFLOPs & GB/s |
| 4 | Intra-node: NVLink / NVSwitch | `nvidia-smi topo -m`; single-node 8-GPU all-reduce; per-link NVLink bandwidth |

### Part II — Inter-node Communication
| # | Layer | Lab (run live) |
| :-- | :--- | :--- |
| 5 | NICs, RDMA, GPUDirect (gVNIC, TCPX/TCPXO/RoCE, plugin model) | Identify the actual data path; enable/compare Fast Socket / TCPX |
| 6 | NCCL collectives (ring vs tree, bus vs algo bandwidth) | `nccl-tests` all-reduce/all-gather sweeps across 2 nodes → plotted GB/s curves |

### Part III — Clustering & Distributed Execution
| # | Layer | Lab (run live) |
| :-- | :--- | :--- |
| 7 | GKE scheduling & topology (device plugin, gang, DWS) | Schedule a 16-GPU gang across both nodes |
| 8 | Job frameworks (JobSet / Kueue, rendezvous) | Multi-node JobSet across 16 GPUs |
| 9 | Distributed training (DDP / FSDP, step → collectives) | 2-node PyTorch job with NCCL tracing; DDP vs FSDP |
| 10 | Observability & debugging (DCGM, NCCL_DEBUG, XID, NIC counters) | Inject faults; read the signatures (hangs, saturation, mismatch) |

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
  part1-single-node/          GPU arch, drivers/CUDA, single-GPU exec, NVLink
  part2-inter-node/           NICs/RDMA/GPUDirect, NCCL collectives
  part3-clustering-execution/ GKE scheduling, JobSet/Kueue, DDP/FSDP, observability
labs/        Step-by-step practice; one dir per lab, with real captured output
manifests/   Reusable, verified Kubernetes YAML
scripts/     Runner + capture/parse scripts
assets/      Captured real outputs: logs, CSVs, plots, diagrams
reference/   Env-var cheat sheets, XID table, NCCL tunables, driver matrix, glossary
VERIFICATION.md   Provenance log of every live run and artifact
```
