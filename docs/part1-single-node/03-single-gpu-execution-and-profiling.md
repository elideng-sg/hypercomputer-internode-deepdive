# 03 — Single-GPU Execution and Profiling

## Overview

This guide covers the **CUDA execution model** — how GPU kernels are launched, scheduled, and executed on a single GPU — and the tools and techniques for **profiling and benchmarking** GPU workloads. Understanding single-GPU performance is the foundation for diagnosing multi-GPU and distributed training bottlenecks.

**What you'll learn:**
- The CUDA execution model: host-device interaction, streams, kernels, and occupancy
- How a training/inference step maps to GPU kernels
- The roofline model: compute-bound vs. memory-bound operations
- Profiling with Nsight Systems (`nsys`) and Nsight Compute (`ncu`)
- Interpreting real profiling data from an H100 GPU
- Memory bandwidth characterization (H2D/D2H)
- Compute throughput measurement (GEMM TFLOPs)

**Hands-on practice:** [lab-03: Single-GPU Benchmark & Profile](../../labs/lab-03-single-gpu-benchmark-profile/)

**Prerequisites:** Basic understanding of GPU architecture (see [doc-01](01-gpu-architecture.md)) and driver/CUDA stack (see [doc-02](02-driver-cuda-health.md)).

---

## The CUDA Execution Model

### Host-Device Interaction

CUDA applications follow a **heterogeneous execution model** where computation is split between:
- **Host (CPU):** Orchestrates the overall workflow, prepares data, launches kernels, manages memory transfers
- **Device (GPU):** Executes massively parallel kernels on thousands of CUDA cores

**Workflow:**
1. **Allocate GPU memory** (`cudaMalloc` or PyTorch `.to(device)`)
2. **Transfer data from host to device** (H2D via PCIe)
3. **Launch kernel** on GPU (asynchronous, returns immediately to host)
4. **GPU executes kernel** (parallel work across SMs)
5. **Transfer results from device to host** (D2H via PCIe)
6. **Synchronize** (`cudaDeviceSynchronize` or `torch.cuda.synchronize()`) to wait for GPU to complete

**Key insight:** Most CUDA operations are **asynchronous**. The host does not wait for GPU completion unless explicitly synchronized. This allows overlapping host work (e.g., data loading, CPU preprocessing) with GPU execution.

### Streams

**Streams** are **independent execution queues** on the GPU. Operations (kernel launches, memory copies) in the same stream execute **sequentially**, while operations in different streams can execute **concurrently** (if hardware resources permit).

**Use cases:**
- **Default stream (stream 0):** Synchronizes with all other streams (serializing behavior)
- **Non-default streams:** Enable concurrent kernel execution (e.g., overlap compute on one stream with memory transfer on another)

**Example (PyTorch):**
```python
stream_a = torch.cuda.Stream()
stream_b = torch.cuda.Stream()

with torch.cuda.stream(stream_a):
    result_a = model_a(input_a)  # Launches kernels on stream A

with torch.cuda.stream(stream_b):
    result_b = model_b(input_b)  # Launches kernels on stream B (can overlap with A)
```

**Why it matters:** Distributed training frameworks (e.g., PyTorch DDP) use multiple streams to **overlap gradient communication with backward-pass computation**, hiding communication latency.

### Kernels

A **kernel** is a function executed on the GPU by many threads in parallel. Threads are organized hierarchically:
- **Grid:** The entire collection of threads for a kernel launch
- **Block:** A group of threads (up to 1024 on modern GPUs) that can cooperate via shared memory and synchronization
- **Warp:** A group of 32 threads that execute in lockstep (SIMT execution)

**Launch configuration:** When launching a kernel, you specify:
- **Grid dimensions** (number of blocks)
- **Block dimensions** (number of threads per block)

**Example:** Matrix multiplication (GEMM) on a 4096×4096 matrix might use:
- Block size: 256 threads (16×16 thread block)
- Grid size: 65536 blocks (256×256 grid)
- Total threads: 16,777,216

Each thread computes one or more output elements.

### Occupancy

**Occupancy** is the ratio of **active warps** to the **maximum possible active warps** on an SM. Higher occupancy generally improves latency hiding (memory stalls can be masked by switching to other warps), but is not always required for peak performance.

**Factors limiting occupancy:**
- **Registers per thread:** Each thread needs GPU registers; if a kernel uses too many, fewer blocks fit on an SM
- **Shared memory per block:** High shared memory usage limits concurrent blocks
- **Thread block size:** Very small blocks underutilize the SM; very large blocks may prevent multiple blocks from running concurrently

**Measuring occupancy:** Use `ncu` (Nsight Compute) to report theoretical and achieved occupancy for a kernel.

---

## How a Training Step Maps to Kernels

A single forward-backward pass in deep learning translates to **hundreds to thousands** of GPU kernel launches. Understanding this mapping is key to interpreting profiler timelines.

### Forward Pass

**Operations:**
1. **Matrix multiplications (GEMM):** Linear layers, attention QKV projections
   - **Kernel family:** cuBLAS GEMM kernels (e.g., `nvjet_hsh_*`, `nvjet_tst_*` on Hopper)
   - **Compute-bound:** High arithmetic intensity, utilizes Tensor Cores
2. **Element-wise operations:** Activations (ReLU, GELU), normalization (LayerNorm, BatchNorm)
   - **Kernel family:** Pointwise kernels (e.g., `at::native::*`)
   - **Memory-bound:** Low arithmetic intensity, bottlenecked by memory bandwidth
3. **Reductions:** Softmax, mean/variance for normalization
   - **Kernel family:** Reduction kernels (e.g., `reduce_sum`, `reduce_max`)
   - **Memory-bound:** Limited by L2/HBM bandwidth

### Backward Pass

**Operations (in reverse order of forward pass):**
1. **Loss gradient computation:** Element-wise gradient of loss
2. **Activation gradients:** Backpropagate through element-wise ops (chain rule)
3. **GEMM gradients:** Compute weight gradients and input gradients
   - **Two GEMMs per forward GEMM:** `dL/dW = input^T @ grad_output`, `dL/dinput = grad_output @ W^T`
4. **Normalization gradients:** High-variance kernels due to sequential dependencies

**Typical ratio:** Backward pass takes **2-3× longer** than forward pass (more GEMMs, more synchronization).

### Optimizer Step

**Operations:**
1. **Gradient aggregation (DDP):** All-reduce to sum gradients across GPUs
2. **Optimizer update:** Apply weight updates (SGD, Adam, etc.)
   - **Kernel family:** Element-wise ops (add, multiply, sqrt for Adam)
   - **Memory-bound:** Streaming reads/writes to weight tensors

---

## The Roofline Model

The **roofline model** is a visual performance model that plots **achieved performance** (TFLOPs or GB/s) against **operational intensity** (FLOPs per byte of memory accessed). It identifies whether a kernel is **compute-bound** or **memory-bound**.

### Key Concepts

**Operational Intensity (OI):**
```
OI = FLOPs / Bytes Accessed
```

High OI = compute-heavy (e.g., large GEMM reusing data in cache)  
Low OI = memory-heavy (e.g., element-wise ops touching every byte once)

**Hardware Peaks:**
- **Compute peak (ridge):** Maximum TFLOPs (varies by dtype: FP8 > FP16 > FP32)
- **Memory bandwidth peak (flat roof):** Maximum GB/s from HBM

**Interpretation:**
- **Below the memory bandwidth line:** Memory-bound (increase OI via fusion, tiling, or reduce data movement)
- **Below the compute line, above memory line:** Compute-bound (optimize kernel efficiency, use Tensor Cores, tune tile sizes)

### H100 Theoretical Peaks

**Compute (dense matrix multiply with Tensor Cores):**
- **FP8 (with FP8 Tensor Cores):** ~2000 TFLOPs (H100 SXM5 spec)
- **FP16/BF16 (with Tensor Cores):** ~1000 TFLOPs
- **TF32 (with Tensor Cores):** ~500 TFLOPs
- **FP32 (CUDA cores):** ~60 TFLOPs
- **FP64 (CUDA cores):** ~30 TFLOPs

**Memory Bandwidth (HBM3):**
- **Theoretical:** 3.35 TB/s (H100 SXM5)
- **Achievable (copy kernels):** ~3.0-3.2 TB/s (due to ECC overhead, memory controller scheduling)

**PCIe Bandwidth (Gen5 x16):**
- **Theoretical:** 128 GB/s (bidirectional)
- **Achievable (pinned memory):** ~60 GB/s per direction (H2D or D2H, not bidirectional simultaneously)

---

## Lab-03 Results: Single-GPU Benchmarking

The following data is from **lab-03** executed on a **NVIDIA H100 80GB HBM3** GPU (A3 High node, GKE cluster).

### GEMM Performance (Measured)

**Test:** Square matrix multiplication (M=N=K) with PyTorch `torch.mm`, 20 iterations after 5 warmup, using CUDA Events for timing.

| Matrix Size | FP16 TFLOPs | BF16 TFLOPs |
|-------------|-------------|-------------|
| 2048        | 645.63      | 631.46      |
| 4096        | 740.24      | 764.58      |
| 8192        | 740.02      | 764.25      |
| 16384       | 651.06      | 688.69      |

**Key observations:**
1. **Peak performance:** ~740-765 TFLOPs for medium-large matrices (4096-8192), achieving **~70-75% of H100 theoretical peak** (~1000 TFLOPs for FP16/BF16)
2. **Small matrix overhead:** 2048×2048 shows lower TFLOPs (~630-645) due to kernel launch overhead and lower parallelism
3. **Large matrix drop:** 16384×16384 shows reduced TFLOPs (~650-690), likely due to memory capacity pressure (16384^2 × 2 bytes × 3 tensors = ~3 GB per GEMM, near L2 cache capacity limits)
4. **BF16 vs FP16:** BF16 slightly outperforms FP16 at larger sizes (better numerical range for accumulation, potentially better Tensor Core scheduling on Hopper)

**Roofline placement:** These GEMMs are **compute-bound** (high operational intensity: ~O(N^3) FLOPs / O(N^2) bytes). Performance is limited by Tensor Core throughput, not memory bandwidth.

**FP8 support:** FP8 data types (`torch.float8_e4m3fn`) are not yet stable in PyTorch 2.x on H100 in this container; test skipped. Production workloads using FP8 (e.g., via Transformer Engine) can achieve closer to the 2000 TFLOPs theoretical peak.

### Memory Bandwidth (H2D/D2H)

**Test:** Host-to-device (H2D) and device-to-host (D2H) transfers using pinned host memory (`pin_memory=True` in PyTorch), 20 iterations per size.

| Size (MB) | H2D (GB/s) | D2H (GB/s) |
|-----------|------------|------------|
| 1         | 21.31      | 2.97       |
| 4         | 25.63      | 2.71       |
| 16        | 26.98      | 2.71       |
| 64        | 27.43      | 1.93       |
| 256       | 27.62      | 1.94       |
| 1024      | 27.66      | 1.95       |

**Key observations:**
1. **H2D bandwidth:** ~27.6 GB/s for large transfers, consistent with **PCIe Gen4 x16 expectations** (~32 GB/s theoretical, minus protocol overhead)
   - Note: A3 High nodes use PCIe Gen4; Gen5 (64 GB/s) would show ~50-60 GB/s
2. **D2H bandwidth:** **Significantly lower (~1.9-3.0 GB/s)**, anomalous compared to typical D2H performance (~20-25 GB/s on PCIe Gen4)
   - **Likely cause:** PyTorch's `.to('cpu')` implementation may involve synchronous operations, additional copies, or memory pinning overhead not present in native CUDA `cudaMemcpy`
   - **Production note:** Use asynchronous CUDA streams with pinned memory for optimal D2H bandwidth

**Takeaway:** For training/inference pipelines, **minimize H2D/D2H transfers**. Batch data transfers and use pinned memory. Ideally, keep data on-device across iterations.

### Nsight Systems Profiling (nsys)

**Command:** `nsys profile -t cuda,nvtx -o gemm python gemm.py`

**NVTX ranges (application-level timing):**
- `GEMM_torch.float16_16384`: 375.9 µs
- `GEMM_torch.bfloat16_16384`: 374.5 µs
- `GEMM_torch.bfloat16_4096`: 367.3 µs
- (8 total NVTX ranges, one per dtype-size combination)

**CUDA API calls (host-side):**
- `cudaDeviceSynchronize`: 728.7 ms total (91.8% of API time) — expected, as the script synchronizes after every timed section
- `cuLibraryLoadData`: 50.8 ms (6.4%) — one-time cuBLAS library loading
- `cudaLaunchKernel`: 7.7 ms (1.0%, 17 calls) — kernel launch overhead (~450 µs avg)

**GPU kernel summary (top 3):**
1. `nvjet_hsh_192x192_64x3_2x1_v_bz_coopB_NNN`: 332.7 ms total, 25 instances, **13.3 ms avg** — **BF16 GEMM kernel**
2. `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN`: 315.6 ms total, 25 instances, **12.6 ms avg** — **FP16 GEMM kernel**
3. `nvjet_hsh_320x128_64x3_1x2_h_bz_coopB_NNT`: 37.4 ms total, 25 instances, **1.5 ms avg** — **BF16 GEMM kernel (smaller size)**

**Key observations:**
- **Two kernel families:** `nvjet_hsh_*` (BF16) and `nvjet_tst_*` (FP16), Hopper-native Tensor Core kernels
- **Kernel time dominates:** 99.7% of execution time is GPU compute (minimal idle time, tight kernel packing)
- **NVTX correlation:** NVTX ranges align with kernel durations, confirming correct instrumentation

**Profiling overhead:** Minimal (<5%) for `nsys` with `-t cuda,nvtx`.

### Nsight Compute (ncu) — Privilege Limitation

**Command attempted:** `ncu --set full -o gemm_ncu python gemm.py`

**Result:** **Failed with permission error**

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

**Cause:** GKE Container-Optimized OS (COS) restricts GPU performance counter access via the NVIDIA kernel module parameter `NVreg_RestrictProfilingToAdminUsers=1`. Nsight Compute requires **CAP_SYS_ADMIN** capability (privileged mode) to access performance counters.

**Workaround (for future labs):** Run the profiling pod with:
```yaml
securityContext:
  privileged: true
```
or
```yaml
securityContext:
  capabilities:
    add:
    - SYS_ADMIN
```

**Contrast with Nsight Systems:** `nsys` uses CUPTI activity tracing (not performance counters), so it **does not require privilege elevation** and works on standard GKE pods.

**Production note:** Reserve `ncu` for offline kernel analysis (kernel replay in a privileged container) or local/DGX development environments. Use `nsys` for system-wide profiling in production.

---

## Roofline Analysis: GEMM on H100

Let's place the measured GEMM performance on the roofline model.

**Assumptions:**
- Matrix size: 8192×8192×8192 (representative mid-large size)
- Data type: BF16 (2 bytes per element)
- Measured performance: 764 TFLOPs

**Operational Intensity Calculation:**

**FLOPs:** `2 × M × N × K = 2 × 8192^3 ≈ 1.1 × 10^12 FLOPs` (per GEMM)

**Bytes accessed (naive model, assuming no cache reuse):**
- Input matrices: `A (8192×8192) + B (8192×8192) = 2 × 8192^2 × 2 bytes = 268 MB`
- Output matrix: `C (8192×8192) = 8192^2 × 2 bytes = 134 MB`
- **Total:** ~402 MB = 4.02 × 10^8 bytes

**Operational Intensity:** `1.1 × 10^12 FLOPs / 4.02 × 10^8 bytes ≈ 2736 FLOPs/byte`

**Roofline comparison:**
- **Memory bandwidth peak:** 3.35 TB/s × 2736 FLOPs/byte = **9170 TFLOPs** (if memory-bound)
- **Compute peak:** 1000 TFLOPs (BF16 Tensor Core spec)

**Achieved:** 764 TFLOPs

**Placement:** The GEMM sits **well into the compute-bound region** (OI = 2736 >> ridge point ~0.3 FLOPs/byte for H100). Performance is limited by Tensor Core throughput, not memory. The achieved 764 TFLOPs is **76.4% of peak**, reasonable given:
- Kernel launch overhead
- Memory alignment / tile size non-optimality
- Hopper's dynamic instruction scheduling (not all Tensor Cores saturated 100% of the time)

**Optimization strategy:** For GEMMs, focus on:
- **Increasing batch size or matrix size** (amortize launch overhead)
- **Using CUDA Graphs** (reduce kernel launch latency)
- **Fusing subsequent element-wise ops** (reduce memory roundtrips)

For **memory-bound ops** (e.g., element-wise activations, normalization), focus on:
- **Kernel fusion** (fuse multiple element-wise ops into one kernel)
- **Memory coalescing** (ensure threads access contiguous memory)
- **Shared memory / L1 cache utilization** (for reused data)

---

## Interpreting Profiler Timelines

When viewing the Nsight Systems timeline (`.nsys-rep` in the GUI):

**Rows to focus on:**
1. **NVTX ranges:** High-level annotations (e.g., "GEMM_torch.bfloat16_8192") — use these to navigate to interesting regions
2. **CUDA HW row:** GPU kernel execution (bars represent kernels) — look for:
   - **Gaps between kernels:** Idle GPU time (investigate why: CPU bottleneck, synchronization, launch overhead)
   - **Kernel duration:** Long kernels are candidates for optimization (zoom in with `ncu`)
3. **CUDA API row:** Host-side CUDA calls — look for:
   - `cudaDeviceSynchronize` or `cudaStreamSynchronize`: Blocking calls (GPU idle waiting for host, or vice versa)
   - Frequent `cudaLaunchKernel` with no corresponding GPU activity: Launch overhead dominating short kernels
4. **Memory Operations:** H2D/D2H/D2D transfers — should overlap with compute (use async copies and streams)

**Common patterns:**
- **Good:** Solid GPU kernel bars with minimal gaps, NCCL collectives overlapping with compute (in multi-GPU)
- **Bad:** Frequent synchronization points causing GPU idle, large H2D/D2H transfers in the critical path

---

## Summary

**Key takeaways:**
1. **CUDA execution model:** Asynchronous kernel launches, streams for concurrency, host-device synchronization points
2. **Training maps to kernels:** Forward/backward/optimizer steps each involve hundreds of kernels (GEMMs, element-wise ops, reductions)
3. **Roofline model:** Distinguishes compute-bound (GEMMs) from memory-bound (element-wise ops) operations
4. **Lab-03 measured performance:**
   - H100 GEMM: **740-765 TFLOPs** (76% of peak) for FP16/BF16
   - PCIe H2D: **~27 GB/s** (PCIe Gen4 x16)
   - PCIe D2H: **~2 GB/s** (anomalously low in PyTorch `.to('cpu')`; native CUDA would be ~20-25 GB/s)
5. **Profiling tools:**
   - **Nsight Systems (`nsys`):** System-wide timeline, no privilege required (use this in production)
   - **Nsight Compute (`ncu`):** Per-kernel metrics, requires CAP_SYS_ADMIN on GKE COS (use offline or in privileged pods)

**Next steps:**
- [doc-04: Multi-GPU and NVLink](04-multi-gpu-nvlink.md) — scaling to multiple GPUs on a single node
- [Toolkit: T3 — Profiling and Tracing](../../docs/toolkit/T3-profiling-tracing.md) — deep dive into `nsys`, `ncu`, NVTX, PyTorch profiler
- [Toolkit: T4 — Benchmarking](../../docs/toolkit/T4-benchmarking.md) — methodology, tools (nvbandwidth, nccl-tests), bus bandwidth vs. algorithm bandwidth

**Hands-on practice:** [lab-03: Single-GPU Benchmark & Profile](../../labs/lab-03-single-gpu-benchmark-profile/)
