# hypercomputer-internode-deepdive

A **standalone, hands-on guide** to the **NVIDIA GPU + AI-infrastructure stack** — from a single GPU's silicon, up through node and cluster communication, to the **purpose-built AI-factory platforms** NVIDIA ships (DGX/HGX, BlueField, Spectrum-X, DGX SuperPOD) — plus the **operations/diagnostics** and **architecture/GCP-integration** skills engineers need to design, run, and debug real workloads. Taught with the **standard NVIDIA toolchain** and backed by **labs run live on real A3 H100 clusters** — a 2-node cluster (16 × H100) and a 3-node cluster (24 × H100) that turns the inter-node "cliff" into a measured scaling *curve*.

One continuous, bottom-up journey, then two applied tracks on top:

**single node → inter-node communication → clustering & distributed execution → platform & reference architectures → operations, diagnostics & troubleshooting → architecture & GCP integration**

…with a **NVIDIA tooling layer** cutting across all of it (nvidia-smi, DCGM, Nsight Systems/Compute, nvbandwidth, nccl-tests, perftest, dcgm-exporter + Grafana, and more).

Every mechanism and tool is paired with a lab capturing **actual measured data** — architecture specs, roofline + profiler timelines, NVLink & inter-node bandwidth curves, topology dumps, NCCL traces, fleet dashboards, and failure signatures — not generic expected values.

> **Scope: the full GCP AI Hypercomputer GPU portfolio — not just A3.** The lab runs live on A3 High (H100), but the guide covers and **attributes each concept to the right GCP product/family** — A3 High (TCPX), A3 Mega (TCPXO), A3 Ultra (H200, RoCE/GPUDirect-RDMA), A4 (B200), A4X (GB200 + NVLink domains) — and maps them to their NVIDIA-platform equivalents: GCP **Titanium** ↔ **BlueField DPU**, GPUDirect-TCPX/RDMA ↔ **Spectrum-X / InfiniBand**, GKE+JobSet+Kueue ↔ **Base Command Manager / Run:ai / Slurm**. Concepts and tools are otherwise **platform-agnostic** (on-prem/DGX/SuperPOD, other clouds, bare metal).

> **Honesty note on Part IV.** DGX/HGX *system software*, BlueField DPUs, and Spectrum-X are **not present** on the GCP A3 cluster (each A3 node rides an **HGX H100 baseboard**, but DGX OS / Fabric Manager / DPUs / Spectrum-X are not tenant-visible). Part IV is therefore **knowledge-first / reference-architecture**, with *observe-and-compare* exercises against what the A3 lab can actually show. It never claims that DGX/BlueField/Spectrum-X hardware was run here.

> **Status: 🚧 in progress.** Live on the A3 clusters. **Captured with real data (Parts I–IV, labs 01–11):** the full toolkit (T1–T6); Part I docs 01–04 + labs 01–04 (incl. the 8-GPU NVLink/NVSwitch mesh and ~480 GB/s intra-node all-reduce ceiling); Part II docs 05–06 + labs 05–06 (the inter-node network path over gVNIC/TCP, then the 2-node 16-GPU NCCL all-reduce, ~28 GB/s — the ~17× inter-node cliff); Part III docs 07–10 + labs 07, 08, 10 (GKE gang scheduling & the DWS Pending gate; JobSet+Kueue admitting a real 4-rank all-reduce — `value=4.0` — and gating an over-quota gang to zero pods; DDP vs FSDP where the step is ~90% all-reduce; and fleet observability via the managed DCGM/GMP pipeline plus four real fault signatures); Part IV docs 11–14 + lab-11 platform-compare.
>
> **Newly captured (3-node scaling, `asia-east1-c`):** **lab-12** + **doc-15** — the 8/16/24-GPU all-reduce **curve** (peak busbw 465 → 23.7 → 14.95 GB/s across 1/2/3 nodes: the inter-node number is a *descending curve*, not a floor) and the ring-vs-tree finding (Tree beats Ring at every message size once ≥3 nodes; NCCL's default under-picks).
>
> **Specced and in build (labs 13–21, Parts V–VI):** the scaling track's resilience lab (lab-13) + DDP/FSDP efficiency; **Part V** operations, diagnostics & troubleshooting (labs 14–17, docs 16–20); **Part VI** architecture & GCP integration (labs 18–21, docs 21–24), including a **live GPUDirect-TCPX before/after**. Design specs live in [`docs/superpowers/specs/`](docs/superpowers/specs/); the three tracks are coordinated by the [integration roadmap](docs/superpowers/specs/2026-07-23-integration-roadmap.md). These are **planned/in-progress and are labelled as such below** until their data is captured.
>
> No fabric or measurement is ever claimed that wasn't read off a live cluster — see [`VERIFICATION.md`](VERIFICATION.md).

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

### Scaling bridge — from cliff to curve  *(3-node cluster)*
Two nodes prove a *cliff*; they cannot show a *curve*. On the 3-node (24 × H100) `asia-east1-c` cluster:
| # | Focus | Lab | Why 2 nodes can't show it |
| :-- | :--- | :--- | :--- |
| 12 | **Captured:** 8/16/24-GPU all-reduce curve (peak busbw 465→23.7→14.95 GB/s; latency floor climbs) + ring-vs-tree (Tree beats Ring at every size at 3 nodes). DDP/FSDP efficiency in build | `lab-12` scaling sweep | The inter-node number is a *descending curve*, not a floor; one point can't show a slope |
| 13 | 24-GPU non-power-of-2 gang placement; Flex-safe job-level node-loss survivor set *(in build)* | `lab-13` topology & resilience | Lose 1 of 2 → 0 survivors; the survivor/elastic story needs N≥3 |
| doc-15 | **Captured:** "Scaling: the shape of the cliff" — the 8/16/24 busbw+latency curve, ring/tree finding, cross-cluster caveat; scaling-efficiency section in build | connective doc | — |

### Part V — Operations, Diagnostics & Troubleshooting  *(scenario-based; specced, in build)*
Every existing lab walks the *healthy path*. Part V is the missing skill: **symptom → hypothesis → tool → read the output → root cause → fix**.
| # | Layer | Lab (scenario) |
| :-- | :--- | :--- |
| 16 | The diagnostic method (triage framework — the hub every scenario links to) | — |
| 17 | Single-GPU & node health (XID on managed GKE, throttling, ECC/RAS, DCGM diag) | `lab-14` single-GPU health triage |
| 18 | Inter-node comms troubleshooting (slow/hung collective, MTU/rail/env faults) | `lab-15` inter-node comms debug |
| 19 | Cluster & job failure triage (OOM, crashloop, stuck gang, Kueue inadmissible) | `lab-16` cluster/job failure triage |
| 20 | Performance monitoring & day-2 ops (baselines, Grafana/PromQL, regression detection) | `lab-17` perf monitoring & day-2 ops |

### Part VI — Architecture & GCP Integration  *(design-first use-cases; specced, in build)*
How a GPU workload is actually **architected and deployed** on GCP — the design skill.
| # | Layer | Lab (design & stand up, then measure) |
| :-- | :--- | :--- |
| 21 | GKE network design (single-gVNIC vs TCPX vs TCPXO vs RDMA as a decision) | `lab-18` **enable GPUDirect-TCPX** — live gVNIC→TCPX before/after (flagship) |
| 22 | Storage & the data path (GCSFuse, Parallelstore, Hyperdisk ML; the starved-GPU) | `lab-19` storage & data-path throughput |
| 23 | End-to-end training pipeline (data → JobSet → checkpoints → metrics; WIF, Artifact Registry) | `lab-20` e2e training pipeline |
| 24 | Inference serving & MLOps (Inference Gateway, autoscale on DCGM metrics, Vertex AI contrast) | `lab-21` inference serving + autoscale |

## The lab environment

Two live A3 High (H100) clusters. Numbers are always labelled by cluster; scaling **curves** are captured on a single cluster (never spliced across clusters).

| | Documented lab cluster | Scaling / 3-node cluster |
| :--- | :--- | :--- |
| Cluster | `hypercomputer-a3-cluster` (GKE v1.33) | `hypercomputer-a3-asiaeast1` (GKE v1.34) |
| Region / zone | `us-central1` | `asia-east1-c` |
| GPU pool | `a3-h100-dws-pool` — 2 × `a3-highgpu-8g` = **16 × H100 80GB** | `a3-high-flex-pool` — 3 × `a3-highgpu-8g` = **24 × H100 80GB** |
| Provisioning | DWS queued, 7-day holds | Flex-start, 7-day expiry (scarce; held) |
| Fabric today | single-gVNIC / TCP (~28.6 GB/s inter-node floor) | single-gVNIC / TCP (Part VI `lab-18` provisions a **new** multi-network TCPX pool to close the cliff) |

**Journey:** single GPU → single node (8×NVLink/HGX) → 2 nodes (the ~17× cliff) → **8/16/24-GPU scaling curve** → 16/24-GPU jobs → platform reference architectures → **operate & troubleshoot** → **architect & integrate on GCP**.

## Design spec

The full design is in
[`docs/superpowers/specs/2026-07-21-hypercomputer-internode-deepdive-design.md`](docs/superpowers/specs/2026-07-21-hypercomputer-internode-deepdive-design.md).

## Repository layout

```
docs/
  00-guide-overview.md            Journey, clusters, portability (GCP lab vs generic)
  toolkit/                        Cross-cutting NVIDIA tooling deep-dives (T1–T6)
  part1-single-node/              GPU arch, drivers/CUDA, single-GPU exec+profiling, NVLink/HGX
  part2-inter-node/               NICs/RDMA/GPUDirect, NCCL collectives
  part3-clustering-execution/     GKE scheduling, JobSet/Kueue, DDP/FSDP+profiling, fleet observability
  part4-platform-reference-arch/  DGX/HGX, BlueField/DOCA, Spectrum-X, DGX SuperPOD
  15-scaling-shape-of-the-cliff.md   Scaling bridge: 8/16/24-GPU curve + ring/tree (captured)
  part5-operations-diagnostics/   Diagnostic method, single-GPU health, comms/cluster/job triage, day-2 ops (in build)
  part6-architecture-gcp-integration/  GKE network design, storage/data path, e2e pipeline, inference serving (in build)
  superpowers/specs/              Design specs + integration roadmap
labs/        Step-by-step practice; one dir per lab, with real captured output (lab-01..12 live; lab-13..21 in build)
manifests/   Reusable, verified Kubernetes YAML (incl. dcgm-exporter, fault injectors, tcpx/storage/serving)
scripts/     Runner + capture/parse scripts (lib_capture.sh, provision_tcpx_pool.sh)
assets/      Captured real outputs: logs, CSVs, plots, profiler timelines, diagrams
reference/   Cheat sheets: XID table, NCCL tunables, driver matrix, tools, reference-arch, glossary
VERIFICATION.md   Provenance log of every live run and artifact
```
