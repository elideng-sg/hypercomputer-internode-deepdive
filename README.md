# hypercomputer-internode-deepdive

A **standalone, hands-on guide** to **how GPUs communicate across nodes** and **how distributed workloads execute** on **Google Cloud AI Hypercomputer** — from NVLink silicon up through NCCL, GKE scheduling, and job frameworks.

Every mechanism is paired with a **lab run live on a real 2-node A3 H100 cluster** (16 × H100 80GB), capturing **actual measured data** — bandwidth curves, topology dumps, NCCL traces, and failure signatures — not generic expected values.

> **Status: 🚧 planning.** The design spec is committed and under review. Implementation (docs + live labs) has not started yet.

---

## What this guide will cover

Layered bottom-up — each layer gets a mechanism deep-dive **and** a hands-on lab:

| # | Layer | Lab (run live) |
| :-- | :--- | :--- |
| 0 | Cluster & topology overview | Enumerate the real cluster: nodes, GPUs, NICs, DaemonSets |
| 1 | Intra-node: NVLink / NVSwitch | `nvidia-smi topo -m`, per-link NVLink bandwidth |
| 2 | Inter-node: NICs, RDMA, GPUDirect | Identify the actual data path; enable/compare Fast Socket / TCPX |
| 3 | NCCL collectives | `nccl-tests` all-reduce/all-gather sweeps → plotted GB/s curves |
| 4 | GKE scheduling & topology | Schedule a 16-GPU gang across both nodes |
| 5 | Job frameworks (JobSet / Kueue) | Multi-node JobSet across 16 GPUs |
| 6 | Distributed training (DDP / FSDP) | 2-node PyTorch job with NCCL tracing |
| 7 | Observability & debugging | Inject faults; read the signatures (XID, hangs, NIC saturation) |

## The lab environment

| | |
| :--- | :--- |
| Cluster | `hypercomputer-a3-cluster` (GKE v1.33), `us-central1` |
| GPU pool | 2 × `a3-highgpu-8g` = **16 × H100 80GB** (DWS) |
| Focus | Inter-node communication + distributed workload execution |

## Design spec

The full design is in
[`docs/superpowers/specs/2026-07-21-hypercomputer-internode-deepdive-design.md`](docs/superpowers/specs/2026-07-21-hypercomputer-internode-deepdive-design.md).

## Repository layout (planned)

```
docs/        Mechanism deep-dives — one file per layer
labs/        Step-by-step practice; one dir per lab, with real captured output
manifests/   Reusable, verified Kubernetes YAML
scripts/     Runner + capture/parse scripts
assets/      Captured real outputs: logs, CSVs, plots, diagrams
reference/   Env-var cheat sheets, XID table, NCCL tunables, glossary
VERIFICATION.md   Provenance log of every live run and artifact
```
