# 01: GPU Microarchitecture — Streaming Multiprocessors, Memory Hierarchy, and Tensor Cores

## Introduction

This document examines the internal architecture of modern datacenter GPUs, focusing on NVIDIA's Hopper H100 — the accelerator powering Google Cloud's A3 machine family (and the hardware under this entire hypercomputer internode deep-dive). Understanding GPU microarchitecture is essential for reasoning about performance bottlenecks, memory access patterns, kernel launch configurations, and multi-GPU scaling behavior. When a collective operation stalls or a kernel hits 10% of expected throughput, the answer often lies in how the workload maps to the GPU's streaming multiprocessors, memory subsystems, and specialized tensor cores.

## CPU vs GPU vs TPU: Design Philosophy

Modern compute accelerators diverge from CPUs in fundamental ways:

**CPUs** optimize for single-thread latency and flexibility. A typical server CPU has 8-64 cores, each running at 2-4 GHz with multi-level caches (MB per core), out-of-order execution, branch prediction, and hardware prefetching. The design invests heavily in minimizing the time from instruction issue to retirement for a single thread. CPUs excel at irregular control flow, pointer chasing, and workloads with unpredictable branches.

**GPUs** optimize for throughput over latency. A datacenter GPU has thousands of CUDA cores organized into 100+ streaming multiprocessors (SMs), each capable of executing warps (groups of 32 threads) in lockstep. Rather than hiding latency with complex out-of-order logic, GPUs hide latency by massively oversubscribing the execution units: when one warp stalls on a memory access, the SM immediately switches to another ready warp. This SIMT (Single Instruction, Multiple Thread) model requires high parallelism and regular memory access patterns to reach peak utilization, but delivers 10-100× the floating-point throughput of a CPU.

**TPUs** (Google's Tensor Processing Units) go further, hardening the matrix-multiply operation into a systolic array with local SRAM and minimizing DRAM round-trips. TPUs sacrifice programmability for maximum efficiency on dense linear algebra (e.g., transformer attention and MLP layers). They are not general-purpose accelerators — irregular workloads or non-matmul kernels fall back to CPU or GPU.

For this series, we focus on GPUs because they balance programmability (CUDA, OpenCL, Triton) with raw throughput, and because A3 nodes are NVIDIA H100-based.

## Streaming Multiprocessors (SMs), SIMT, and Warps

The GPU's execution model centers on the **streaming multiprocessor (SM)**. Each SM is an independent processing block containing:

- **CUDA Cores** (FP32 and INT32 ALUs): Execute scalar arithmetic operations. A Hopper SM has 128 FP32 units.
- **Tensor Cores**: Specialized units for matrix operations (covered below).
- **Special Function Units (SFUs)**: Accelerate transcendentals (sin, cos, exp, etc.).
- **Load/Store Units (LD/ST)**: Issue memory requests to the cache hierarchy.
- **Warp Schedulers**: Select warps for execution each cycle.

A **warp** is a group of 32 threads that execute the same instruction in lockstep (SIMT = Single Instruction, Multiple Threads). When a CUDA kernel launches, the GPU groups threads into warps. All threads in a warp execute the same instruction on different data elements. If threads in a warp take different code paths (e.g., due to an `if` statement), the warp must serialize the paths (**branch divergence**), executing each path while masking off inactive threads. This serialization can halve (or worse) the effective throughput.

**Key takeaway:** GPU performance depends on high occupancy (many warps resident per SM) and warp-level uniformity (all threads in a warp following the same control flow and accessing contiguous memory).

## Memory Hierarchy: Registers → Shared → L2 → HBM3

A GPU's memory subsystem is explicitly hierarchical:

1. **Registers** (per thread): Fastest storage, zero-latency access. Limited capacity (255 registers per thread max). Spilling registers to local memory (backed by DRAM) destroys performance.

2. **Shared Memory / L1 Cache** (per SM): Fast scratchpad (typically 100-200 KB per SM). Shared memory is programmer-managed; adjacent threads can cooperate by loading data into shared memory, synchronizing with `__syncthreads()`, and reusing it. Shared memory has low latency (~20 cycles) and high bandwidth, but limited capacity. Efficient use requires tiling algorithms and coalesced access patterns.

3. **L2 Cache** (global, shared across all SMs): Large pool (tens of MB) with lower latency than HBM but higher than L1/shared. Automatically managed by hardware. Effective L2 locality is critical for memory-bound kernels.

4. **HBM (High Bandwidth Memory)** (off-chip DRAM): The main device memory pool. Hopper H100 uses **HBM3**, delivering 80 GB capacity and ~3 TB/s bandwidth. HBM latency is 200-400 cycles. Kernels that repeatedly round-trip to HBM (e.g., element-wise operations on large tensors) become bandwidth-bound.

**Memory access efficiency** hinges on **coalescing**: when threads in a warp access consecutive memory addresses, the hardware coalesces their requests into a minimal number of cache-line transactions. Strided or random accesses fragment requests and waste bandwidth.

## Tensor Cores: Matrix Multiply Accelerators

Modern datacenter GPUs dedicate significant die area to **Tensor Cores** — specialized units that execute small matrix multiplies (e.g., D = A × B + C, where A, B, C, D are 16×16 or 8×8 tiles) in a single instruction. Tensor Cores deliver 8-20× the throughput of standard FP32 CUDA cores for dense linear algebra.

Hopper H100 Tensor Cores support:

- **FP8** (8-bit floating point): Highest throughput, suitable for training and inference with mixed-precision recipes. H100 delivers ~2000 TFLOPS (tera-floating-point-operations per second) for FP8 matmuls.
- **BF16** (bfloat16): 16-bit format with FP32 exponent range. Standard for training transformers. ~1000 TFLOPS on H100.
- **TF32** (TensorFloat-32): NVIDIA's FP32-compatible format (FP32 input, lower mantissa precision). Default for CUDA matmuls. ~500 TFLOPS.
- **FP64** (double precision): For scientific computing. ~60 TFLOPS.
- **INT8**: For quantized inference. ~2000 TOPS (tera-operations per second).

To leverage Tensor Cores, code must call cuBLAS, cuDNN, or use higher-level frameworks (PyTorch, JAX) with the right dtypes. Manually-written kernels can invoke `mma` (matrix-multiply-accumulate) PTX instructions or use Cutlass/Triton abstractions.

## Hopper H100 Specifics

NVIDIA's Hopper architecture (compute capability 9.0) introduces several improvements over Ampere (A100):

1. **Fourth-generation Tensor Cores**: FP8 support, higher throughput, transformer engine (automatic FP8/BF16 casting).
2. **Thread Block Clusters**: Kernels can organize thread blocks into clusters that share an extended L1 cache and synchronize across blocks. Useful for tiling large matmuls.
3. **TMA (Tensor Memory Accelerator)**: Asynchronous bulk data transfer engine between global and shared memory, reducing pipeline stalls.
4. **Larger L2 Cache**: 50 MB L2 (vs 40 MB on A100).
5. **NVLink 4.0**: 900 GB/s bidirectional per GPU for multi-GPU scaling (discussed in Part 2).

### What Our H100 Actually Reports

From `assets/lab-01/smi-q.txt` and `assets/lab-01/devquery.txt`:

| Property                 | Value                                           | Source                           |
| :----------------------- | :---------------------------------------------- | :------------------------------- |
| Product Name             | NVIDIA H100 80GB HBM3                          | `smi-q.txt` line 11              |
| Architecture             | Hopper                                          | `smi-q.txt` line 13              |
| Compute Capability       | 9.0                                             | `devquery.txt` line 9            |
| Streaming Multiprocessors| 132                                             | `devquery.txt` line 11           |
| CUDA Cores               | 16896 (132 SMs × 128 cores/SM)                  | `devquery.txt` line 11           |
| Max SM Clock             | 1980 MHz                                        | `smi-q.txt` line 199             |
| Memory Total             | 81559 MiB (~80 GB usable)                       | `smi-q.txt` line 96              |
| Memory Clock             | 2619 MHz                                        | `smi-q.txt` line 188             |
| Memory Bus Width         | 5120-bit                                        | `devquery.txt` line 14           |
| L2 Cache Size            | 50 MiB                                          | `devquery.txt` line 15           |
| Shared Memory per Block  | 48 KiB                                          | `devquery.txt` line 20           |
| Registers per Block      | 65536                                           | `devquery.txt` line 21           |
| Max Threads per SM       | 2048                                            | `devquery.txt` line 23           |
| TDP                      | 700 W                                           | `smi-q.txt` line 173             |
| ECC                      | Enabled                                         | `smi-q.txt` line 125             |

**Note:** 132 SMs is the full H100 80GB die count. This is the configuration used in Google Cloud's `a3-highgpu-8g` machine type (8×H100 per node). The memory bandwidth calculation: 2619 MHz × 5120-bit bus ÷ 8 bits/byte = ~3.35 TB/s theoretical (actual measured bandwidth is ~3.0 TB/s accounting for ECC and protocol overhead).

## Hopper vs Blackwell (A3 vs A4)

Google Cloud's **A3** machine family uses Hopper H100 GPUs. The next generation, **A4**, will use NVIDIA's Blackwell architecture (B100/B200). Blackwell doubles FP8 Tensor Core throughput (>4000 TFLOPS), increases NVLink bandwidth to 1.8 TB/s per GPU, and integrates a second die in a dual-chip module. For this series, we focus on A3/Hopper — the hardware available today.

## Summary

GPUs achieve high throughput by massively parallel SIMT execution (warps of 32 threads on 100+ SMs), explicit memory hierarchy (registers → shared → L2 → HBM3), and specialized Tensor Cores for matrix operations. The Hopper H100 delivers 132 SMs, 80 GB HBM3, ~3 TB/s memory bandwidth, and up to 2000 TFLOPS for FP8 matmuls. Understanding this architecture is the foundation for reasoning about single-node performance before scaling to multi-node collectives.

**Practice →** [lab-01: GPU Architecture Inspection](../../labs/lab-01-gpu-arch-inspect/README.md)  
**Tools in this layer →** [T1: Monitoring & Inventory](../toolkit/T1-monitoring-inventory.md)
