# Lab 04: Intra-node NVLink / NVSwitch (HGX H100 baseboard)

**Objective:** Measure the **intra-node** GPU interconnect — the full 8-GPU NVLink4 / NVSwitch mesh on the A3's HGX H100 baseboard — and establish the single-node NCCL all-reduce bandwidth ceiling. Every inter-node number in Part II is measured against this ceiling.

**Duration:** ~10 minutes (inside an existing 8-GPU pod)

**Prerequisites:**
- An 8-GPU pod on one A3 node (this run used `nccl-workbench-a`; the `nccl-tests` binaries ship in `nvcr.io/nvidia/pytorch:24.10-py3`)
- `REPO_ROOT` set, `scripts/lib_capture.sh` sourced
- Read [doc-04](../../docs/part1-single-node/04-intranode-nvlink-nvswitch-hgx.md)

---

## Run

```bash
LAB04_POD=nccl-workbench-a bash labs/lab-04-intranode-nvlink-hgx/run.sh
```

**GPU safety:** the script does not allocate GPUs itself — it `kubectl exec`s into a pod you already have holding 8 GPUs. In this environment `hv7m` is normally held by a DWS capacity holder; the full 8-GPU run was performed during a planned holder-repurpose window and the holder was re-armed immediately afterward.

---

## What was measured (real output)

### 1. Full 8-GPU NV18 all-to-all mesh — `assets/lab-04/topo-8gpu.txt`

`nvidia-smi topo -m` across all 8 GPUs shows every GPU pair connected by **NV18** — 18 bonded NVLink4 links through the on-baseboard NVSwitches:

```
      GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0   X    NV18  NV18  NV18  NV18  NV18  NV18  NV18
GPU1  NV18   X    NV18  NV18  NV18  NV18  NV18  NV18
 ...   (all off-diagonal cells are NV18) ...
GPU7  NV18  NV18  NV18  NV18  NV18  NV18  NV18   X
```

This is the defining property of an HGX baseboard: **uniform, non-blocking all-to-all** — GPU0↔GPU7 has the same 18-link path as GPU0↔GPU1. There is no PCIe or NUMA hop between any GPU pair (contrast the `CPU Affinity` column: GPUs 0–3 are on NUMA 0, 4–7 on NUMA 1, yet the GPU-to-GPU path is still NVLink, not QPI).

### 2. Per-link rate → ~900 GB/s bidirectional per GPU — `assets/lab-04/nvlink-status.txt`

`nvidia-smi nvlink --status` reports **18 links × 26.562 GB/s** for GPU 0:

```
Link 0: 26.562 GB/s
 ... (Links 0–17, all 26.562 GB/s) ...
Link 17: 26.562 GB/s
```

18 × 26.562 ≈ **478 GB/s per direction**, ≈ **956 GB/s bidirectional** per GPU — the NVLink4 spec for Hopper.

### 3. Fabric trained by the (invisible) Fabric Manager — `assets/lab-04/fabric-state.txt`

```
Fabric
    State   : Completed
    Status  : Success
```

The NVSwitch fabric is trained and healthy. As lab-11 showed, the tenant sees this *result* but cannot run `nv-fabricmanager` — Google manages it below the tenant boundary.

### 4. Single-node 8-GPU all-reduce sweep — `assets/lab-04/all_reduce_8gpu.txt`

`all_reduce_perf -b 8 -e 8G -f 2 -g 8` (NCCL 2.22.3+cuda12.6), busbw vs. message size:

| size | busbw (GB/s) |
| ---: | ---: |
| 1 MB | 46.2 |
| 16 MB | 233.5 |
| 128 MB | 395.2 |
| 1 GB | 464.4 |
| 8 GB | **479.9** |

- **Peak busbw ≈ 480 GB/s** at 8 GiB — the intra-node NVLink ceiling for this 8-GPU ring.
- Small messages (8 B–64 KB) are **latency-bound** at ~33–35 µs — NCCL launch + sync overhead, not bandwidth.
- The knee is around 4–16 MB; beyond ~256 MB the curve is flat at the fabric ceiling.

> **Why < 956 GB/s?** `busbw` for a ring all-reduce is the *effective* rate each GPU sustains while every GPU is simultaneously sending and receiving `2(n-1)/n` × the data. ~480 GB/s sustained across an 8-GPU ring is the expected NVLink-bound result; it is **not** the per-link peak.

---

## The number that matters for Part II

**Intra-node all-reduce peak: ~480 GB/s.** Lab-06 measures the *same* collective across **two** nodes and lands at ~28 GB/s — a **~17× drop** the moment the ring must cross the node boundary over Ethernet instead of NVLink. That gap is the entire motivation for GPUDirect-TCPX/TCPXO, RoCE, and InfiniBand.

---

**Concepts →** [doc-04 Intra-node NVLink/NVSwitch](../../docs/part1-single-node/04-intranode-nvlink-nvswitch-hgx.md) · [doc-01 GPU microarchitecture](../../docs/part1-single-node/01-gpu-microarchitecture.md) · [§11 DGX/HGX](../../docs/part4-platform-reference-arch/11-dgx-hgx-systems.md)
**Contrast →** [lab-06 2-node NCCL](../lab-06-2node-nccl-collectives/) · [lab-11 platform compare](../lab-11-platform-compare/)
**Tools →** [T4 Benchmarking](../../docs/toolkit/T4-benchmarking.md) · [T5 Networking/Fabric](../../docs/toolkit/T5-networking-fabric-tools.md)
