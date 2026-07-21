# T4 — GPU and Collective Benchmarking

## Overview

This reference document covers the tools, methodology, and interpretation techniques for benchmarking GPUs and collective communication primitives on NVIDIA GPU clusters. Benchmarking is central to **validating performance expectations**, **diagnosing underperformance**, and **characterizing new hardware** before production workloads land.

The guide is organized into two parts:
1. **Methodology** — the foundational principles of correct benchmarking (warmup, repeats, bandwidth definitions, clock control)
2. **Tools** — the canonical NVIDIA and community benchmarking tools, their flags, output formats, and interpretation

All examples reference the live GKE A3 H100 cluster used throughout this guide; labs lab-03, lab-04, and lab-06 provide hands-on practice with these tools.

---

## Part 1: Benchmarking Methodology

Benchmarking GPUs and GPU-to-GPU communication requires attention to several measurement hygiene principles. Skipping these steps produces misleading or non-reproducible results.

### 1.1 Warmup Iterations

**Why:** The first execution of a GPU kernel or collective operation incurs one-time costs:
- CUDA context initialization
- GPU clock ramp-up (power management transitions from idle to active P-states)
- NCCL topology discovery and connection establishment (ring/tree algorithm setup)
- Page table setup and TLB misses
- Instruction cache misses

**Practice:** Run at least **3–5 warmup iterations** before measuring. Discard their timing results. Most NVIDIA benchmarks (e.g., `nccl-tests`) do this automatically; for custom microbenchmarks, implement it explicitly.

### 1.2 Repeated Measurements

**Why:** GPU and network performance is subject to variance from:
- Competing workloads (CPU scheduling, network traffic from other tenants)
- Thermal throttling (dynamic clock frequency adjustments)
- PCIe link state transitions
- NCCL algorithm selection (NCCL may choose different ring/tree configurations across runs)

**Practice:** Run each test **at least 10–20 iterations** and report the **median** (not mean). The median is robust to outliers caused by transient interference. For production characterization, consider running 100+ iterations.

**Reporting:**
- **Median:** The 50th percentile; preferred for summarizing typical performance.
- **Mean:** Arithmetic average; useful if you need to model total runtime over many executions, but sensitive to outliers.
- **Min/Max:** Report these alongside the median to show variance. A large gap suggests instability (thermal throttling, network contention, or preemption).

Example:
```
Size: 1 GB, iterations: 20
  Min:  8.2 GB/s
  Median: 9.1 GB/s
  Mean: 8.9 GB/s
  Max: 9.3 GB/s
```

### 1.3 Clock Frequency Control

**Why:** Modern GPUs dynamically adjust their clock frequencies based on:
- Power consumption
- Temperature
- Workload type (compute-bound vs. memory-bound)

This means two runs of the same benchmark may execute at different clock speeds, producing different throughput numbers.

**Practice:** For reproducible results, lock the GPU clocks to a known frequency using `nvidia-smi`:

```bash
# Query available application clocks
nvidia-smi -q -d SUPPORTED_CLOCKS

# Lock clocks (requires root or CAP_SYS_ADMIN)
nvidia-smi -lgc <graphics_clock_mhz>
nvidia-smi -lmc <memory_clock_mhz>

# Example: lock H100 to max application clocks
nvidia-smi -lgc 1980
nvidia-smi -lmc 2619

# Reset to default (auto) behavior
nvidia-smi -rgc
nvidia-smi -rmc
```

**Caution:** Clock locking requires elevated privileges. In Kubernetes, this typically means running in a privileged Pod or using a DaemonSet with `hostPID: true` and `securityContext.privileged: true`. GKE A3 clusters allow this; verify your environment's security policy.

**When to lock clocks:**
- Comparing hardware configurations (A100 vs. H100, different node types)
- Debugging performance regressions (isolate clock variance from code changes)
- Establishing vendor-comparable peak numbers (NVIDIA spec sheets assume max clocks)

**When not to lock clocks:**
- Characterizing production workloads (you want to see real-world behavior, including thermal effects)
- Power efficiency studies (dynamic clocks save power on bursty workloads)

### 1.4 Bus Bandwidth vs. Algorithm Bandwidth

This is the **most critical** and **most commonly misunderstood** distinction in collective benchmarking.

#### Definitions

**Algorithm Bandwidth (algbw):**  
The effective rate at which **data is logically moved** from the perspective of the collective operation's semantics.

For an **all-reduce** with `n` GPUs and message size `S`:
- Each GPU **sends** `S` bytes and **receives** `S` bytes (the reduced result).
- Total data movement per GPU: `S` bytes.
- If the operation takes time `T`, then:
  ```
  algbw = S / T
  ```

**Bus Bandwidth (busbw):**  
The effective rate of **physical data movement on the interconnect** (NVLink, PCIe, network), accounting for the collective algorithm's redundancy.

For a **ring all-reduce** with `n` GPUs:
- The ring algorithm performs `n-1` steps.
- In each step, every GPU sends a chunk and receives a chunk.
- Total physical data sent across all links: `n × (n-1) × (S/n) = (n-1) × S`.
- Each GPU contributes `(n-1) × S` of link utilization.
- If the operation takes time `T`, then:
  ```
  busbw = (n-1) × S / T
  ```

**Relationship:**
```
busbw = algbw × (n-1)
```

Or, in the form most tools report:
```
busbw = algbw × (2(n-1) / n)
```
This second form accounts for **bidirectional** link utilization in some contexts (see note below).

#### Why busbw is the hardware-comparable number

When comparing collective performance to **hardware specifications** (e.g., NVLink 900 GB/s bidirectional per H100, or 400 Gb/s RoCE NIC line rate), you must use **busbw**, not algbw.

**Example:**  
- 8-GPU all-reduce, ring algorithm, message size 1 GB, time 10 ms.
- `algbw = 1 GB / 0.01 s = 100 GB/s`
- `busbw = 100 GB/s × 7 = 700 GB/s`

The `algbw` (100 GB/s) represents the **logical data movement rate per GPU**. The `busbw` (700 GB/s) represents the **aggregate link utilization** and is what you compare against the NVLink fabric's peak bidirectional bandwidth.

#### Derivation of the (2(n-1)/n) factor

Some tools (including `nccl-tests`) define `busbw` as:
```
busbw = algbw × (2(n-1) / n)
```

This formula accounts for **bidirectional simultaneous send/receive** on each link in a ring all-reduce:
- Each GPU simultaneously **sends** and **receives** in each of the `n-1` steps.
- The factor of 2 counts both directions.
- The factor of `(n-1)/n` normalizes by the number of GPUs.

**In practice:** Most modern tools report `busbw` directly in their output. Always check the tool's documentation to confirm which definition it uses. When in doubt, compare the reported `busbw` against known hardware limits to sanity-check the factor.

#### Practical interpretation

| Metric | Use Case |
| :--- | :--- |
| **algbw** | Application-level throughput; how fast data is collectively reduced |
| **busbw** | Hardware utilization; compare to NVLink/NIC specs to diagnose bottlenecks |

**When benchmarking:**
1. Run the benchmark and record both `algbw` and `busbw`.
2. Compare `busbw` to the hardware's **bidirectional peak bandwidth**.
3. If `busbw` is significantly lower than expected, investigate:
   - Topology (is NCCL using the optimal ring/tree?)
   - Contention (are multiple collectives running simultaneously?)
   - Configuration (NCCL tuning variables, CUDA graphs, pinned memory)

### 1.5 Message Size Sweeps

Collective and memory bandwidth performance is **highly size-dependent**:
- **Small messages** (< 1 KB): latency-bound; dominated by kernel launch overhead and protocol headers.
- **Medium messages** (1 KB – 1 MB): transition region; bandwidth grows as amortization kicks in.
- **Large messages** (> 1 MB): bandwidth-bound; asymptotically approaches peak hardware bandwidth.

**Practice:** Always sweep across **at least 3 orders of magnitude** of message sizes. Use a geometric progression (powers of 2 or factors of 2):
```
Sizes: 8 B, 32 B, 128 B, 512 B, 2 KB, 8 KB, 32 KB, 128 KB, 512 KB, 2 MB, 8 MB, 32 MB, 128 MB, 512 MB, 1 GB, 4 GB
```

Most NVIDIA benchmarks support `-b <min>`, `-e <max>`, and `-f <factor>` flags to automate this sweep (see `nccl-tests` below).

---

## Part 2: Benchmarking Tools

### 2.1 nvbandwidth

**Source:** [NVIDIA/nvbandwidth](https://github.com/NVIDIA/nvbandwidth)  
**Purpose:** Measure memory and interconnect bandwidth for GPU↔host, GPU↔GPU, and host↔device paths.

`nvbandwidth` is the **canonical tool** for characterizing:
- **Host-to-Device (H2D):** PCIe write bandwidth (CPU pinned memory → GPU)
- **Device-to-Host (D2H):** PCIe read bandwidth (GPU → CPU pinned memory)
- **Device-to-Device (P2P):** NVLink or PCIe P2P bandwidth between GPUs

#### Installation

On the lab cluster, `nvbandwidth` is available in the CUDA toolkit container images (e.g., `nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04`). To build from source:

```bash
git clone https://github.com/NVIDIA/nvbandwidth.git
cd nvbandwidth
make
```

#### Running nvbandwidth

**Full test suite (all test cases):**
```bash
./nvbandwidth
```

**Specific test cases:**

Each test case is identified by a numeric ID. Common test IDs:

| Test ID | Description |
| :--- | :--- |
| `testcase=1` | Host-to-Device (H2D) write, all GPUs in parallel |
| `testcase=2` | Device-to-Host (D2H) read, all GPUs in parallel |
| `testcase=3` | Device-to-Device (P2P) unidirectional, all pairs |
| `testcase=4` | Device-to-Device (P2P) bidirectional, all pairs |
| `testcase=5` | Host-to-Device (H2D) and Device-to-Host (D2H) simultaneously |

**Example: measure P2P bandwidth between all GPU pairs:**
```bash
./nvbandwidth -t testcase=3
```

**Example: measure bidirectional P2P bandwidth:**
```bash
./nvbandwidth -t testcase=4
```

**Example: measure H2D and D2H on GPU 0:**
```bash
CUDA_VISIBLE_DEVICES=0 ./nvbandwidth -t testcase=1
CUDA_VISIBLE_DEVICES=0 ./nvbandwidth -t testcase=2
```

#### Interpreting Output

Output is a matrix of bandwidth values in GB/s. For P2P tests:
```
     GPU0   GPU1   GPU2   GPU3   GPU4   GPU5   GPU6   GPU7
GPU0   -    275.3  275.1  275.0  268.2  267.9  268.0  267.8
GPU1 275.2   -    275.1  275.0  268.1  267.8  268.0  267.9
...
```

**Key checks:**
- **Within-node NVLink (H100):** Expect ~270–280 GB/s unidirectional per link (NVLink 4.0 is rated at 900 GB/s bidirectional per GPU pair, divided across the HGX baseboard topology).
- **PCIe Gen4 x16:** Expect ~24–26 GB/s (theoretical max 32 GB/s minus protocol overhead).
- **Asymmetry:** If GPU0→GPU1 and GPU1→GPU0 differ significantly, investigate topology (`nvidia-smi topo -m`) or NVLink errors (`nvidia-smi nvlink -e`).

**Practice:** See lab-03 for hands-on `nvbandwidth` execution and interpretation on the A3 cluster.

### 2.2 CUDA-samples bandwidthTest

**Source:** [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples), under `Samples/1_Utilities/bandwidthTest`  
**Purpose:** Measure host↔device memory transfer bandwidth (PCIe) and device-to-device bandwidth.

`bandwidthTest` is a simpler, single-binary alternative to `nvbandwidth` for basic PCIe and P2P characterization. It's useful for quick checks but lacks the matrix-sweep detail of `nvbandwidth`.

#### Installation

Available in NVIDIA CUDA sample containers or build from source:
```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/bandwidthTest
make
```

#### Running bandwidthTest

**Default mode (H2D, D2H, D2D):**
```bash
./bandwidthTest
```

**Flags:**
- `--htod`: Host-to-Device only
- `--dtoh`: Device-to-Host only
- `--dtod`: Device-to-Device only
- `--device=N`: Test GPU N
- `--memory=pinned|pageable`: Use pinned or pageable host memory (pinned is default and faster)

**Example: measure H2D and D2H on GPU 0:**
```bash
./bandwidthTest --htod --dtoh --device=0
```

#### Interpreting Output

Typical output:
```
Host to Device Bandwidth, 1 Device(s)
 Transfer Size (Bytes)  Bandwidth(GB/s)
 32000000               26.4

Device to Host Bandwidth, 1 Device(s)
 Transfer Size (Bytes)  Bandwidth(GB/s)
 32000000               26.1
```

**Key checks:**
- H2D/D2H on PCIe Gen4 x16: ~24–26 GB/s (pinned memory).
- H2D/D2H on PCIe Gen3 x16: ~12–13 GB/s.
- Significantly lower values suggest PCIe link degradation, competing workloads, or pageable memory usage.

### 2.3 nccl-tests

**Source:** [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)  
**Purpose:** Measure **collective communication** performance (all-reduce, all-gather, reduce-scatter, broadcast, etc.) using the NCCL library.

`nccl-tests` is the **canonical benchmark** for collective operations. It directly exercises the same NCCL code paths that PyTorch DDP, TensorFlow Horovod, and other distributed training frameworks use in production.

#### Installation

Build from source (requires NCCL installed or in the container):
```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1  # if MPI is available; omit for single-node tests
```

On GKE A3, use the NCCL-enabled containers from NGC:
```bash
docker pull nvcr.io/nvidia/pytorch:24.07-py3
# nccl-tests is pre-built in /opt/nccl-tests/build/
```

#### Test Binaries

Each collective has its own binary:
- `all_reduce_perf`: All-reduce
- `all_gather_perf`: All-gather
- `reduce_scatter_perf`: Reduce-scatter
- `broadcast_perf`: Broadcast
- `reduce_perf`: Reduce
- `alltoall_perf`: All-to-all

#### Common Flags

| Flag | Description |
| :--- | :--- |
| `-b <min_bytes>` | Minimum message size (default: 8 bytes) |
| `-e <max_bytes>` | Maximum message size (default: 256 MB) |
| `-f <factor>` | Multiplicative factor for size sweep (default: 2) |
| `-g <gpus_per_proc>` | Number of GPUs per MPI process (default: 1) |
| `-n <iters>` | Number of iterations per size (default: 20) |
| `-w <warmup_iters>` | Number of warmup iterations (default: 5) |
| `-c <check>` | Data correctness check (0=off, 1=on; default: 1) |
| `-t <threads>` | Number of GPUs per process (synonym for `-g`) |
| `-m <min_size>` | Synonym for `-b` |
| `-M <max_size>` | Synonym for `-e` |

#### Running nccl-tests

**Single-node, 8 GPUs:**
```bash
./all_reduce_perf -b 8 -e 4G -f 2 -g 1 -n 20
```
- Sweep from 8 bytes to 4 GB, doubling each step.
- 1 GPU per process (8 processes total, launched by the test harness).
- 20 iterations per size.

**Multi-node with MPI (example: 2 nodes, 8 GPUs each = 16 GPUs total):**
```bash
mpirun -np 16 --hostfile hostfile \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_DISABLE=0 \
  ./all_reduce_perf -b 1K -e 1G -f 2 -g 1
```

In Kubernetes (GKE), use `mpirun` wrapper Jobs or JobSets with `torchrun`-style launchers; see lab-04 and lab-06 for manifests.

#### Interpreting Output

Typical output columns:
```
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    25.73    0.00    0.00      0    25.48    0.00    0.00      0
          16             4     float     sum      -1    25.90    0.00    0.00      0    25.65    0.00    0.00      0
          32             8     float     sum      -1    26.18    0.00    0.00      0    25.89    0.00    0.00      0
      ...
    1048576        262144     float     sum      -1    141.3    7.42   12.98      0    139.7    7.51   13.14      0
    2097152        524288     float     sum      -1    203.2   10.32   18.05      0    201.8   10.39   18.18      0
  268435456      67108864     float     sum      -1   8735.0   30.74   53.79      0   8698.0   30.87   54.02      0
 1073741824     268435456     float     sum      -1  33421.0   32.13   56.23      0  33312.0   32.23   56.41      0
```

**Columns explained:**
- `size (B)`: Message size in bytes.
- `count (elements)`: Number of elements (size / sizeof(datatype)).
- `type`: Data type (`float`, `half`, `int`, etc.).
- `redop`: Reduction operation (`sum`, `prod`, `max`, `min`).
- `root`: Root rank for rooted collectives (all-reduce is `-1`, meaning no root).
- `time (us)`: **Median** time in microseconds (across `-n` iterations).
- `algbw (GB/s)`: **Algorithm bandwidth** = `size / time`.
- `busbw (GB/s)`: **Bus bandwidth** = `algbw × (2(n-1)/n)` for all-reduce with `n` GPUs.
- `#wrong`: Number of data correctness errors (should always be 0).

**Out-of-place vs. in-place:**
- **Out-of-place:** Separate send and receive buffers.
- **In-place:** Same buffer for send and receive (saves memory).

Performance is typically similar; report the **in-place** numbers for production workloads (most frameworks use in-place by default).

#### Key Checks

**Single-node (8 × H100, NVLink):**
- **Large messages (≥128 MB):** Expect `busbw` approaching **400–450 GB/s** (NVLink 4.0 bidirectional aggregate across the HGX baseboard).
- **Small messages (≤4 KB):** Latency-bound; `algbw` will be low (<1 GB/s), and `time` is the key metric (~25–30 µs per step for NVLink).

**Multi-node (inter-node network):**
- **A3 High (GPUDirect-TCPX, gVNIC):** Expect `busbw` of **80–120 GB/s** for large messages (depends on network congestion and NCCL tuning).
- **A3 Ultra (RoCE, CX-7 400 Gb/s):** Expect `busbw` approaching **180–220 GB/s** for large messages (400 Gb/s ≈ 50 GB/s per NIC × 8 NICs × ring factor).

Compare your results against these baselines in lab-04 (single-node) and lab-06 (multi-node).

#### NCCL Environment Variables

`nccl-tests` honors all NCCL tuning environment variables. Key ones for benchmarking:

| Variable | Purpose |
| :--- | :--- |
| `NCCL_DEBUG=INFO` | Verbose logging (topology, algorithm selection, errors) |
| `NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV` | Fine-grained debug subsystems |
| `NCCL_IB_DISABLE=0` | Enable InfiniBand/RoCE (if available) |
| `NCCL_P2P_LEVEL=NVL` | Force NVLink for intra-node (default auto-detects) |
| `NCCL_ALGO=Ring` or `Tree` | Force algorithm (for debugging; default is auto) |
| `NCCL_NET_GDR_LEVEL=5` | Enable GPUDirect RDMA (if supported) |

See `docs/reference/nccl-tuning-guide.md` for a comprehensive list.

**Practice:** See lab-04 (single-node all-reduce) and lab-06 (multi-node all-reduce) for hands-on `nccl-tests` execution.

### 2.4 cuBLAS and cuDNN Microbenchmarks

**Purpose:** Measure **compute throughput** (FP32/FP16/INT8 TFLOPs) for matrix multiplication (GEMM) and deep learning primitives (convolution, pooling, normalization).

#### cuBLAS GEMM Benchmark

cuBLAS does not ship a standalone benchmark, but the community maintains simple wrappers. Example using `nvidia-smi` and a PyTorch script:

**Python (PyTorch):**
```python
import torch
import time

# Setup
device = torch.device("cuda:0")
dtype = torch.float16  # or torch.float32, torch.bfloat16
M, N, K = 8192, 8192, 8192
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)

# Warmup
for _ in range(10):
    C = torch.mm(A, B)
torch.cuda.synchronize()

# Measure
iters = 100
start = time.time()
for _ in range(iters):
    C = torch.mm(A, B)
torch.cuda.synchronize()
elapsed = time.time() - start

# Compute TFLOPs
flops = 2 * M * N * K * iters  # 2 FLOPs per multiply-add
tflops = flops / elapsed / 1e12
print(f"GEMM ({M}x{N}x{K}, {dtype}): {tflops:.2f} TFLOPs")
```

**Key checks (H100 80GB):**
- **FP16 Tensor Core GEMM:** Expect **800–1000 TFLOPs** for large matrices (with Tensor Cores enabled).
- **FP32 GEMM:** Expect **60–80 TFLOPs** (CUDA cores only, no Tensor Cores).
- **FP8 GEMM (Hopper):** Expect **2000–3000 TFLOPs** (new in H100).

Use `nvidia-smi dmon -s u` during the run to confirm GPU utilization is >95%.

#### cuDNN Benchmark

NVIDIA ships cuDNN samples with convolution benchmarks. Clone [NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend) and run the `conv_sample`:

```bash
git clone https://github.com/NVIDIA/cudnn-frontend.git
cd cudnn-frontend/samples
mkdir build && cd build
cmake .. && make
./conv_sample
```

Alternatively, use PyTorch's built-in convolution and measure TFLOPs similarly to the GEMM example above.

**Key checks:**
- Convolution throughput scales with batch size and channel count.
- Expect **50–90% of peak GEMM TFLOPs** for typical ResNet-50 or EfficientNet workloads.

### 2.5 gpu-burn

**Source:** [wilicc/gpu-burn](https://github.com/wilicc/gpu-burn)  
**Purpose:** Stress-test GPU compute throughput and stability by running continuous GEMM operations.

`gpu-burn` is **not a precise benchmark** but a **stability and thermal validation tool**. It saturates the GPU at 100% utilization and detects hardware faults (ECC errors, thermal throttling, bad memory).

#### Installation

```bash
git clone https://github.com/wilicc/gpu-burn.git
cd gpu-burn
make
```

#### Running gpu-burn

**Default (run until manually stopped):**
```bash
./gpu_burn
```

**Run for a fixed duration (e.g., 60 seconds):**
```bash
./gpu_burn 60
```

**Monitor all GPUs:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

#### Interpreting Output

`gpu-burn` prints:
```
GPU 0: OK (temperature: 72C)
GPU 1: OK (temperature: 71C)
...
```

**Key checks:**
- All GPUs should report `OK`.
- Temperatures should stabilize below **85°C** (H100 max is 90°C; check `nvidia-smi --query-gpu=temperature.gpu.max --format=csv`).
- No ECC errors (`nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv`).
- GPU clocks should not throttle (`nvidia-smi --query-gpu=clocks.current.graphics --format=csv` should remain at or near max application clocks).

**If `gpu-burn` reports `FAULTY`:**
- Check `nvidia-smi` for ECC errors, thermal throttling (`clocks_throttle_reasons.hw_thermal_slowdown`), or XID errors in `dmesg`.
- See `docs/toolkit/T2-diagnostics-health.md` for troubleshooting steps.

**Practice:** Use `gpu-burn` as a pre-deployment validation step (run for 10–60 minutes) to catch faulty GPUs before production workloads land. See lab-03 for an example.

---

## Part 3: Practical Workflow

### Benchmarking Checklist

When benchmarking a new node or cluster, follow this order:

1. **Inventory:** Confirm GPU model, count, and topology (`nvidia-smi topo -m`).
2. **Health check:** Run `dcgmi diag -r 3` (see `docs/toolkit/T2-diagnostics-health.md`) to verify all GPUs pass extended diagnostics.
3. **Single-GPU memory bandwidth:** Run `nvbandwidth` testcase 1 and 2 (H2D/D2H) to establish PCIe baseline.
4. **Intra-node GPU-to-GPU bandwidth:** Run `nvbandwidth` testcase 3 and 4 (P2P) to confirm NVLink is functioning and measure peer-to-peer bandwidth.
5. **Single-node collective:** Run `nccl-tests` `all_reduce_perf` with 8 GPUs, sweep 8 B to 4 GB, to establish intra-node collective baseline.
6. **Multi-node collective:** Run `nccl-tests` `all_reduce_perf` with 2+ nodes to characterize inter-node network bandwidth.
7. **Compute throughput:** Run cuBLAS GEMM microbenchmark or `gpu-burn` for 60 seconds to confirm peak TFLOPs and thermal stability.
8. **Document:** Record all results with timestamps, cluster state, and NCCL environment variables in your lab notebook or `VERIFICATION.md`.

### Comparing Results

When comparing benchmarks across configurations:
- **Lock clocks** (see 1.3) or document clock frequencies at the time of measurement.
- **Control NCCL variables:** Use identical `NCCL_*` environment variables across runs.
- **Report median, not mean:** Use the median from 20+ iterations.
- **Compare busbw to hardware specs:** Use `busbw` (not `algbw`) to diagnose hardware bottlenecks.

### Troubleshooting Poor Performance

If benchmarks underperform:

| Symptom | Likely Cause | Diagnostic Steps |
| :--- | :--- | :--- |
| Low `busbw` on single-node all-reduce | NVLink degraded or disabled | `nvidia-smi topo -m`, `nvidia-smi nvlink -e` (check for link errors), `NCCL_DEBUG=INFO` (check if NCCL falls back to PCIe) |
| Low `busbw` on multi-node all-reduce | Network congestion or misconfiguration | `nccl-tests` with `NCCL_DEBUG=INFO`, check for `NET/Socket` fallback instead of `NET/IBext`, verify GPUDirect RDMA is enabled |
| Low GEMM TFLOPs | Thermal throttling or incorrect data type | `nvidia-smi dmon -s t` (check temperature), confirm Tensor Cores are enabled (FP16/BF16), lock clocks |
| High variance (large min/max gap) | Competing workloads or thermal cycling | Run on an idle node, lock clocks, increase iteration count |

See `docs/toolkit/T2-diagnostics-health.md` and `docs/part2-inter-node/06-nccl-collectives.md` for deeper troubleshooting.

---

## Summary

Correct GPU and collective benchmarking requires:
1. **Warmup and repeated measurements** (report median).
2. **Clock frequency control** (lock clocks for reproducibility).
3. **Understanding bus bandwidth vs. algorithm bandwidth** (use `busbw` to compare against hardware specs).
4. **Message size sweeps** (small, medium, large).
5. **The right tool for the job:**
   - `nvbandwidth`: H2D/D2H/P2P memory bandwidth.
   - `nccl-tests`: Collective communication (all-reduce, all-gather).
   - cuBLAS/cuDNN microbenchmarks: Compute throughput (TFLOPs).
   - `gpu-burn`: Stability and thermal validation.

The math behind `busbw`:
```
busbw = algbw × (2(n-1) / n)   for ring all-reduce with n GPUs
```
This factor accounts for the redundancy in the ring algorithm and bidirectional link utilization. Always compare `busbw` (not `algbw`) to hardware specs.

---

## Practice

Apply these tools and methods in the following labs:
- **lab-03:** Single-GPU and intra-node bandwidth (`nvbandwidth`, `bandwidthTest`, single-node `nccl-tests`).
- **lab-04:** Single-node all-reduce (`nccl-tests all_reduce_perf`, 8 GPUs).
- **lab-06:** Multi-node all-reduce (`nccl-tests` across 2 nodes, 16 GPUs, GPUDirect-TCPX characterization).

Each lab provides step-by-step instructions, manifests, and result interpretation tied to the live GKE A3 H100 cluster.
