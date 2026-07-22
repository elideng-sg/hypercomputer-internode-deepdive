# 06 — NCCL Collectives and the Inter-node Bandwidth Floor

## Overview

This is the pivot of the whole guide: the point where a collective's ring leaves the ~480 GB/s NVLink world of a single baseboard (doc-04) and crosses onto the network. We measure a real **all-reduce across two A3 nodes (16 GPUs)**, read the **actual NCCL transport** off the wire, and derive why the number lands where it does. Everything in Part II about NICs, RDMA, and GPUDirect exists to raise this floor.

**What you'll learn:**
- What NCCL **collectives** are (all-reduce, all-gather, reduce-scatter) and why training is built on them
- The **ring** and **tree** algorithms, and the `busbw = algbw · 2(n-1)/n` bus-bandwidth definition
- How to read the **NCCL transport log** (`NET/IB`, `NET/Socket`, `GPU Direct RDMA`) to prove the data path
- The measured **~28 GB/s inter-node floor** on this (TCP-only) cluster and the **~17× cliff** vs. intra-node
- Why the transport — not the algorithm — is the ceiling here

**Hands-on practice:** [lab-06: 2-node NCCL collectives](../../labs/lab-06-2node-nccl-collectives/)

**Prerequisites:** the intra-node ceiling ([doc-04](../part1-single-node/04-intranode-nvlink-nvswitch-hgx.md)); benchmarking / bus-vs-algorithm bandwidth ([T4](../toolkit/T4-benchmarking.md)). The physical NIC path (gVNIC, GPUDirect-TCPX, RDMA) is [doc-05](05-nic-rdma-gpudirect.md).

---

## Collectives: the vocabulary of distributed training

A **collective** is a communication primitive over a *group* of ranks (here, 16 GPUs). Three carry almost all of distributed training:

*Figure: all-reduce over 4 ranks — before, each rank holds its own value; after, every rank holds the sum.*

```mermaid
graph LR
  subgraph B1["Before"]
    p0["R0: a"]
    p1["R1: b"]
    p2["R2: c"]
    p3["R3: d"]
  end
  subgraph A1["After: every rank holds the sum"]
    q0["R0: a+b+c+d"]
    q1["R1: a+b+c+d"]
    q2["R2: a+b+c+d"]
    q3["R3: a+b+c+d"]
  end
  p0 --> q0
  p1 --> q1
  p2 --> q2
  p3 --> q3
  classDef ctx fill:#e8eaed,stroke:#9aa0a6,color:#202124;
  classDef good fill:#188038,stroke:#0d652d,color:#ffffff;
  class p0,p1,p2,p3 ctx
  class q0,q1,q2,q3 good
```

*Figure: all-gather — each rank contributes one shard; after, all ranks hold the full tensor.*

```mermaid
graph LR
  subgraph B2["Before: each rank owns one shard"]
    g0["R0: A"]
    g1["R1: B"]
    g2["R2: C"]
    g3["R3: D"]
  end
  subgraph A2["After: every rank owns all shards"]
    h0["R0: ABCD"]
    h1["R1: ABCD"]
    h2["R2: ABCD"]
    h3["R3: ABCD"]
  end
  g0 --> h0
  g1 --> h1
  g2 --> h2
  g3 --> h3
  classDef ctx fill:#e8eaed,stroke:#9aa0a6,color:#202124;
  classDef good fill:#188038,stroke:#0d652d,color:#ffffff;
  class g0,g1,g2,g3 ctx
  class h0,h1,h2,h3 good
```

*Figure: reduce-scatter — sum across ranks, then each rank keeps a single result shard.*

```mermaid
graph LR
  subgraph B3["Before: each rank has the full vector"]
    s0["R0: a,b,c,d"]
    s1["R1: a,b,c,d"]
    s2["R2: a,b,c,d"]
    s3["R3: a,b,c,d"]
  end
  subgraph A3["After: each keeps one summed shard"]
    t0["R0: sum a"]
    t1["R1: sum b"]
    t2["R2: sum c"]
    t3["R3: sum d"]
  end
  s0 --> t0
  s1 --> t1
  s2 --> t2
  s3 --> t3
  classDef ctx fill:#e8eaed,stroke:#9aa0a6,color:#202124;
  classDef good fill:#188038,stroke:#0d652d,color:#ffffff;
  class s0,s1,s2,s3 ctx
  class t0,t1,t2,t3 good
```

| Collective | What it does | Where it shows up |
| :--- | :--- | :--- |
| **all-reduce** | sum a tensor across all ranks, every rank gets the result | DDP gradient sync (doc-09) |
| **all-gather** | each rank contributes a shard, all ranks get the full tensor | FSDP forward (materialize params) |
| **reduce-scatter** | sum across ranks, each rank keeps one shard of the result | FSDP backward (shard grads) |

NCCL (NVIDIA Collective Communications Library) implements these. It picks an **algorithm** (ring, tree) and a **protocol** (Simple, LL, LL128) per collective, per message size, per topology — you saw it choose `AllReduce_Sum_f32_TREE_LL` for the 2-node shape in lab-09.

### Ring all-reduce and the `busbw` definition

The classic algorithm is the **ring**: `n` ranks in a logical loop, the tensor split into `n` chunks, and `2(n-1)` steps (a reduce-scatter phase + an all-gather phase). Each rank sends and receives `(n-1)/n` of the tensor in each phase.

*Figure: ring all-reduce — n ranks in a loop run a reduce-scatter phase then an all-gather phase, 2(n-1) steps total.*

```mermaid
graph LR
  R0["Rank 0"] --> R1["Rank 1"]
  R1 --> R2["Rank 2"]
  R2 --> R3["Rank 3"]
  R3 --> R0
  R0 -.-> ph1["Phase 1: reduce-scatter<br/>n-1 steps"]
  ph1 --> ph2["Phase 2: all-gather<br/>n-1 steps"]
  ph2 --> note["2(n-1) steps, each moves (n-1)/n<br/>busbw = algbw x 2(n-1)/n"]
  classDef meas fill:#1a73e8,stroke:#0b57d0,color:#ffffff;
  classDef good fill:#188038,stroke:#0d652d,color:#ffffff;
  classDef accent fill:#f9ab00,stroke:#b06000,color:#202124;
  class ph1 meas
  class ph2 good
  class note accent
```

That factor is exactly why we report **bus bandwidth** (`busbw`) rather than raw **algorithm bandwidth** (`algbw = message_size / time`):

```
busbw = algbw · 2(n-1)/n
```

`busbw` normalizes out the collective's inherent traffic multiplier so numbers are comparable across different rank counts and against the hardware link rate. (Full derivation in [T4](../toolkit/T4-benchmarking.md).) Both lab-04 and lab-06 use this identical definition, which is what makes the 480-vs-28 comparison fair.

### Ring vs. tree

*Figure: ring (bandwidth-optimal, latency grows with n) vs. tree (log-depth, latency-optimal) — NCCL picks TREE for the 2-node shape.*

```mermaid
graph TD
  subgraph Ring["Ring: bandwidth optimal, latency ~ n"]
    r0["R0"] --> r1["R1"]
    r1 --> r2["R2"]
    r2 --> r3["R3"]
    r3 --> r0
  end
  subgraph Tree["Tree: log depth, latency optimal"]
    t0["R0"] --> t1["R1"]
    t0 --> t2["R2"]
    t1 --> t3["R3"]
  end
  classDef good fill:#188038,stroke:#0d652d,color:#ffffff;
  classDef meas fill:#1a73e8,stroke:#0b57d0,color:#ffffff;
  class r0,r1,r2,r3 good
  class t0,t1,t2,t3 meas
```

- **Ring** maximizes bandwidth (every link busy) but latency grows with `n`.
- **Tree** (log-depth) minimizes latency for small messages and few nodes — which is why NCCL selected TREE for the 2-node/16-rank all-reduce. On a bandwidth-starved link, algorithm choice matters far less than the link itself, as the numbers below show.

---

## Read the transport, don't assume it

The single most important habit in inter-node work: **make NCCL tell you the data path.** Set `NCCL_DEBUG=INFO` (optionally `NCCL_DEBUG_SUBSYS=INIT,NET`) and read the init lines. From lab-06 (`assets/lab-06/nccl_transport.txt`, verbatim):

```
NCCL INFO NCCL version 2.22.3+cuda12.6
NCCL INFO NET/IB : No device found.
NCCL INFO NET/Socket : Using [0]eth0:10.128.0.52<0>
NCCL INFO Using network Socket
NCCL INFO NET/Socket : GPU Direct RDMA Disabled for HCA 0 'eth0'
NCCL INFO Channel 00/16 :  0  7  6  5  4  3  2  1  8 15 14 13 12 11 10  9
```

*Figure: the 16-GPU ring — 14 NVLink hops within the two nodes, 2 slow Ethernet/gVNIC hops crossing the boundary (staged GPU to host to NIC, no GPUDirect RDMA).*

```mermaid
graph LR
  subgraph NodeA["Node A: GPUs 0-7 (NVLink)"]
    a0["rank 0"]
    achain["ranks 7..2<br/>7 NVLink hops"]
    a1["rank 1"]
    a0 --- achain
    achain --- a1
  end
  subgraph NodeB["Node B: GPUs 8-15 (NVLink)"]
    b8["rank 8"]
    bchain["ranks 15..10<br/>7 NVLink hops"]
    b9["rank 9"]
    b8 --- bchain
    bchain --- b9
  end
  a1 =="Ethernet / gVNIC<br/>GPU to host to NIC"==> b8
  b9 =="Ethernet / gVNIC<br/>NIC to host to GPU"==> a0
  a1 -.-> stage["No GPUDirect RDMA:<br/>every crossing byte stages<br/>GPU to host RAM to NIC"]
  classDef good fill:#188038,stroke:#0d652d,color:#ffffff;
  classDef crit fill:#c5221f,stroke:#7a161c,color:#ffffff;
  class a0,achain,a1,b8,bchain,b9 good
  class stage crit
  linkStyle 4 stroke:#c5221f,stroke-width:3px;
  linkStyle 5 stroke:#c5221f,stroke-width:3px;
```

Line by line:

- **`NET/IB : No device found`** — NCCL looked for an InfiniBand / RDMA verbs device and found none.
- **`Using network Socket` + `eth0`** — it fell back to its **TCP socket** transport over the standard **gVNIC** (`10.128.0.x`). There is **no GPUDirect-TCPX plugin** on this cluster; if there were, you'd see `NET/GPUDirectTCPX` here instead.
- **`GPU Direct RDMA Disabled`** — data cannot move NIC↔GPU directly; every inter-node byte stages **GPU → host RAM → NIC → … → NIC → host RAM → GPU**. Those extra copies are latency and bandwidth you pay on every collective.
- **`Channel 00/16`** — the 16-GPU ring. Ranks 0–7 are on node A, 8–15 on node B; the ring crosses the node boundary at `1→8` and `9→0`. **14 of 16 hops are NVLink; 2 are Ethernet** — and those 2 set the pace for all 16.

> This is the honesty discipline of the whole repo: we don't *claim* a fabric, we read it off NCCL's own init log and paste it in.

---

## The measurement: ~28 GB/s, and why

lab-06's all-reduce sweep (16 ranks, 2 nodes, NCCL 2.22.3), `busbw` vs. size:

| Message size | `algbw` (GB/s) | `busbw` (GB/s) | Regime |
| ---: | ---: | ---: | :--- |
| 8 B | ~0.00 | ~0.00 | latency floor ~**257 µs** |
| 1 MB | 2.18 | 4.09 | ramping |
| 16 MB | 7.58 | 14.21 | ramping |
| 128 MB | 11.70 | 21.93 | near-ceiling |
| 512 MB | 15.19 | 28.48 | ceiling |
| 1 GB | 15.26 | **28.60** | ceiling |

**Peak busbw ≈ 28.6 GB/s.** Two features to read:

1. **Latency floor ≈ 257 µs** for tiny messages — vs. ~34 µs single-node (doc-04). Crossing the node boundary is **~7.6× higher latency** before bandwidth even enters the picture. This is why frameworks bucket/fuse small tensors before communicating (doc-09).
2. **Bandwidth ceiling ≈ 28.6 GB/s** — set by the two TCP hops in the ring, and consistent with plain TCP over a single gVNIC flow with host-staging (no GPUDirect). This is *not* the algorithm's fault: NCCL is doing the right thing; the pipe is the limit.

### The ~17× cliff

| all-reduce, 1 GB | Fabric | Peak `busbw` |
| :--- | :--- | ---: |
| 8-GPU, single node (doc-04 / lab-04) | NVLink4 / NVSwitch | **~480 GB/s** |
| 16-GPU, two nodes (this doc / lab-06) | TCP over gVNIC | **~28.6 GB/s** |

**≈ 17× slower** the instant the ring crosses a node boundary. This single fact is the reason the rest of Part II and Part IV exist. The ladder for closing the gap:

- **GPUDirect-TCPX** (A3 High) — NIC↔GPU DMA over TCP, NCCL plugin; multiple NICs/rails.
- **GPUDirect-TCPXO** (A3 Mega) — the optimized successor, higher achievable bandwidth.
- **GPUDirect-RDMA / RoCE** (A3 Ultra, A4) — RDMA verbs, `NET/IB` device present, `GPU Direct RDMA Enabled`.
- **InfiniBand + SHARP** (reference DGX SuperPOD, [§14](../part4-platform-reference-arch/14-dgx-superpod.md)) — in-network reduction offloads the all-reduce itself.

On *this* cluster none are enabled, so we honestly measure the TCP floor. When you run the same lab on a TCPX/RDMA-enabled cluster, the transport line changes and the ~28 GB/s rises accordingly — the lab is written to make that change self-evident.

---

## Summary

**Key takeaways:**
1. Distributed training rides three collectives — **all-reduce** (DDP), **all-gather** + **reduce-scatter** (FSDP) — implemented by NCCL over ring/tree algorithms.
2. Always report **`busbw = algbw·2(n-1)/n`** so numbers compare across rank counts and against link rates.
3. **Read the transport** (`NCCL_DEBUG=INFO`): this cluster shows `NET/Socket`/`eth0`, `NET/IB: No device found`, `GPU Direct RDMA Disabled` — **plain TCP over gVNIC**.
4. Measured inter-node all-reduce peaks at **~28.6 GB/s** with a **~257 µs** latency floor — a **~17× cliff** below the intra-node ceiling.
5. That cliff is a **transport** limit, not an algorithm limit; GPUDirect-TCPX/TCPXO/RDMA/SHARP exist to close it.

**Next steps:**
- [doc-05: NICs, RDMA, GPUDirect](05-nic-rdma-gpudirect.md) — the physical NIC path and what each acceleration tier changes
- [doc-09: Distributed training — DDP & FSDP](../part3-clustering-execution/09-distributed-training-ddp-fsdp.md) — this collective inside a real training step (89.63% of GPU time)
- [§13: Spectrum-X and fabrics](../part4-platform-reference-arch/13-spectrum-x-and-fabrics.md) — the Ethernet-fabric side of closing the cliff

**Hands-on practice:** [lab-06: 2-node NCCL collectives](../../labs/lab-06-2node-nccl-collectives/)
