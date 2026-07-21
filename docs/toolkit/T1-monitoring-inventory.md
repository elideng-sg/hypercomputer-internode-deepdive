# T1: Monitoring & Inventory — nvidia-smi, NVML, DCGM

## What this is

This reference covers the foundational layer of GPU monitoring and inventory tools in the NVIDIA stack: **nvidia-smi** (command-line interface), **NVML** (NVIDIA Management Library, the programmatic API underlying nvidia-smi), and **DCGM** (Data Center GPU Manager, the enterprise-grade telemetry and diagnostics framework). Together these tools provide the baseline observability needed to inspect GPU hardware properties, track runtime utilization and health metrics, diagnose issues, and feed metrics pipelines for fleet-scale monitoring.

### Scope

For each tool, this doc explains:
- **What it measures:** The specific hardware and runtime metrics exposed
- **When to use it:** The scenarios and workflows where the tool is appropriate
- **How to invoke it:** Exact command syntax with flags, programmatic API patterns, or architecture overview

This is a **reference doc**, not a hands-on tutorial. Real command outputs and interpretation workflows are captured in the labs (see "Practice" at the end of this doc).

---

## nvidia-smi: The GPU Swiss Army Knife

**nvidia-smi** (NVIDIA System Management Interface) is the primary command-line tool for querying GPU properties, monitoring real-time metrics, and controlling device state. It's a thin wrapper around NVML (see next section) and is present on every NVIDIA GPU host (bare metal, VM, container, or Kubernetes pod with GPU access).

### What it measures

nvidia-smi exposes:
- **Inventory:** GPU model, driver version, CUDA version, bus ID, UUID
- **Utilization:** GPU compute and memory usage percentages
- **Memory:** Total, used, and free device memory (HBM)
- **Clocks:** Current and maximum SM, memory, and graphics clocks
- **Power:** Current power draw, power limit, power management state
- **Temperature:** GPU and memory junction temperatures
- **Processes:** List of processes using each GPU with their memory consumption
- **Topology:** GPU-to-GPU and GPU-to-NIC interconnect topology (NVLink, PCIe)
- **Error counts:** ECC (Error-Correcting Code) single-bit and double-bit errors, retired pages
- **Throttle reasons:** Thermal, power, or HW slowdown events

### When to use it

- **Quick health check:** `nvidia-smi` (default table) shows at-a-glance status of all GPUs on a host
- **Interactive debugging:** Query detailed state when a GPU job fails or hangs
- **Process inspection:** Identify which PIDs are using which GPUs
- **Topology discovery:** Map out GPU-to-GPU NVLink connections and GPU-to-NIC affinity before running distributed workloads
- **Spot checks during development:** Monitor utilization and memory usage while iterating on CUDA code or ML training scripts

nvidia-smi is **not** ideal for:
- Continuous monitoring at scale (use DCGM + Prometheus for that)
- Profiling kernel-level performance (use `nsys` or `ncu`)
- Diagnosing hardware failures (use `dcgmi diag`)

### Commands

#### Default table view

```bash
nvidia-smi
```

Displays a summary table with one row per GPU: index, name, temperature, power, memory usage, utilization, and running processes. This is the starting point for any GPU inspection.

#### Full query mode

```bash
nvidia-smi -q
```

Dumps the complete state of all GPUs in a multi-section text format (inventory, utilization, clocks, power, temperature, ECC, processes, topology). Use this to capture a full snapshot for troubleshooting or to grep for specific fields.

#### Targeted query with specific sections

```bash
nvidia-smi -q -d MEMORY,UTILIZATION,CLOCK,POWER,ECC,TEMPERATURE
```

Queries only the specified sections, reducing output verbosity. Useful when you know which metric you're investigating (e.g., `-d ECC` to check for memory errors, `-d POWER` to diagnose throttling).

#### Device monitoring (streaming table)

```bash
nvidia-smi dmon
```

Continuously samples and prints a compact table of GPU metrics at 1-second intervals (configurable with `-c` and `-d` flags): SM utilization, memory utilization, encoder/decoder usage, power, and temperature. Useful for observing GPU behavior over time during a workload run.

Example output schema (columns):
```
# gpu   pwr  gtemp  mtemp     sm    mem    enc    dec   mclk   pclk
#   W      C      C      %      %      %      %    MHz    MHz
```

#### Process monitoring (streaming table)

```bash
nvidia-smi pmon
```

Continuously samples and prints a table of per-process GPU usage: PID, GPU index, SM utilization, memory utilization, encoder/decoder usage, and process name. Useful for tracking which processes are using the GPU and how much.

Example output schema (columns):
```
# gpu        pid  type    sm   mem   enc   dec   command
#                           %     %     %     %
```

#### Topology matrix

```bash
nvidia-smi topo -m
```

Displays the interconnect topology between GPUs and between GPUs and NICs (or other devices) as a matrix. Each cell shows the connection type: `SYS` (PCIe via QPI), `PHB` (PCIe via PCIe host bridge, same root complex), `PIX` (single PCIe switch), `NV#` (# NVLink connections), `NODE` (NUMA crossing), or `X` (unsupported).

This command is critical for understanding:
- Which GPUs are directly connected via NVLink vs. PCIe
- GPU-to-NIC affinity (which NICs are "close" to which GPUs for GPUDirect RDMA)
- Whether your workload's GPU placement will incur NUMA or root-complex crossings

Example use case: Before running a multi-GPU NCCL benchmark, check the topology to confirm all 8 GPUs in an HGX baseboard are fully connected via NVLink (you should see `NV12` or `NV18` depending on generation).

#### Filtering and formatting

```bash
# Query specific GPU by index
nvidia-smi -i 0 -q

# Query specific GPU by UUID
nvidia-smi -i GPU-<uuid> -q

# CSV output (for scripting)
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

# Loop mode (legacy alternative to dmon)
nvidia-smi -l 1
```

### nvidia-smi gotchas

- **Utilization sampling:** The default table's "Volatile GPU-Util" is a **coarse-grained rolling average** over the past ~1 second. It may show 0% even when kernels are running if they complete very quickly or if they're memory-bound (SM idle). Use `dmon` for finer-grained streaming, or `nsys` for true kernel-level timeline profiling.
- **Memory accounting:** `nvidia-smi` shows **allocated device memory**, not *used* memory. A process may allocate a large buffer but only touch a fraction of it. The number reflects what's reserved from the GPU's HBM pool.
- **Driver version vs. CUDA version:** The "CUDA Version" shown in the header is the **maximum CUDA runtime version supported by the driver**, not the version your application is using. Your app's actual CUDA version may be lower (forward-compatible).
- **ECC overhead:** On ECC-enabled GPUs (all datacenter GPUs), ~6–7% of the advertised HBM capacity is reserved for ECC metadata and is not visible in the "Total" memory reported by nvidia-smi.

---

## NVML: The Programmatic API

**NVML** (NVIDIA Management Library) is the C library that nvidia-smi is built on. It provides programmatic access to the same inventory and monitoring metrics, plus control APIs (e.g., setting persistence mode, resetting clocks, ECC configuration). NVML is part of the NVIDIA driver and ships with every GPU installation.

### What it is

NVML is a **C API** exposed via `libnvidia-ml.so` (Linux) or `nvml.dll` (Windows). It's intended for:
- Custom monitoring daemons and dashboards
- Integration into HPC or ML job schedulers
- Automated health checks and alerting pipelines
- Tools like DCGM, `nvtop`, `gpustat`, and `dcgm-exporter` (all use NVML under the hood)

nvidia-smi is essentially a command-line frontend to NVML. Anything nvidia-smi can do, you can do via NVML in your own code.

### When to use it

- You need to **embed GPU monitoring** into a custom application or script
- You want to **poll metrics in a tight loop** with sub-second intervals
- You need **control operations** not exposed by nvidia-smi (e.g., setting accounting mode, advanced ECC queries)
- You're building a tool that needs to be **responsive** (NVML calls are fast, <1 ms for most queries)

Do NOT use NVML for:
- Fleet-scale telemetry collection (use DCGM, which batches NVML calls efficiently and handles per-GPU locking)
- Complex diagnostics or health checks (use `dcgmi diag`)

### Python example with pynvml

The most common way to use NVML outside of C is via **pynvml**, the official Python bindings. Here's a minimal example:

```python
import pynvml

# Initialize NVML
pynvml.nvmlInit()

# Get device count
device_count = pynvml.nvmlDeviceGetCount()
print(f"Found {device_count} GPU(s)")

# Query first GPU
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
name = pynvml.nvmlDeviceGetName(handle)
uuid = pynvml.nvmlDeviceGetUUID(handle)
temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(f"GPU 0: {name}")
print(f"  UUID: {uuid}")
print(f"  Temp: {temp}°C")
print(f"  Utilization: {util.gpu}% (SM), {util.memory}% (Mem)")
print(f"  Memory: {mem_info.used / 1024**3:.2f} GB / {mem_info.total / 1024**3:.2f} GB")

# Shutdown NVML
pynvml.nvmlShutdown()
```

Installation:
```bash
pip install pynvml
```

**Key functions** (selection):
- `nvmlDeviceGetHandleByIndex(index)` / `nvmlDeviceGetHandleByUUID(uuid)`: Get a device handle
- `nvmlDeviceGetName()`, `nvmlDeviceGetUUID()`, `nvmlDeviceGetSerial()`: Inventory
- `nvmlDeviceGetTemperature()`, `nvmlDeviceGetPowerUsage()`, `nvmlDeviceGetClockInfo()`: Telemetry
- `nvmlDeviceGetUtilizationRates()`, `nvmlDeviceGetMemoryInfo()`: Utilization
- `nvmlDeviceGetComputeRunningProcesses()`: Process list
- `nvmlDeviceGetTopologyCommonAncestor()`: Topology queries

Full API reference: [NVIDIA NVML Documentation](https://docs.nvidia.com/deploy/nvml-api/)

### NVML vs. nvidia-smi vs. DCGM

| Tool | Use case | Invocation | Performance | Scope |
| :--- | :--- | :--- | :--- | :--- |
| **nvidia-smi** | Interactive CLI inspection | Command-line | ~100–500 ms per call (slow, designed for human use) | Single host, ad hoc |
| **NVML** | Programmatic monitoring | C/Python/Go API | <1 ms per metric | Single host, custom apps |
| **DCGM** | Fleet-scale telemetry, diagnostics | DCGM API, `dcgmi` CLI, `dcgm-exporter` | Batched, agent-based, <10 ms for multi-GPU queries | Single host or cluster, production monitoring |

**Rule of thumb:** Use nvidia-smi for humans, NVML for custom integration, DCGM for production monitoring at scale.

---

## DCGM: Data Center GPU Manager

**DCGM** (Data Center GPU Manager) is NVIDIA's enterprise-grade telemetry, health, and diagnostics framework for multi-GPU, multi-node environments. It's built on NVML but adds:
- **Agent-based architecture** with a persistent daemon (`nv-hostengine`)
- **Efficient batch sampling** of hundreds of metrics across multiple GPUs
- **Health checks and stress tests** (`dcgmi diag`)
- **Field groups** for organizing related metrics
- **Prometheus integration** via `dcgm-exporter`
- **Per-job accounting and profiling** (DCGM Profiling API)

DCGM is the monitoring foundation for production AI clusters (DGX SuperPOD, GKE A3/A4, etc.) and is the recommended tool for **continuous, fleet-scale GPU observability**.

### Architecture

DCGM has a client-server architecture:

1. **nv-hostengine** (daemon): Runs on each GPU host, polls NVML for metrics at configurable intervals (default 1 s), caches values, and exposes them via a gRPC API or shared memory.
2. **dcgmi** (CLI client): Command-line tool for querying metrics, running diagnostics, and controlling the hostengine.
3. **dcgm-exporter** (Prometheus exporter): Scrapes DCGM metrics from `nv-hostengine` and exposes them in Prometheus format for ingestion by Prometheus/Grafana/etc.
4. **DCGM API** (C/Python/Go): Programmatic interface for embedding DCGM into custom applications or orchestrators.

On Kubernetes (GKE, etc.), `nv-hostengine` is typically deployed as a **DaemonSet** (one pod per GPU node) and `dcgm-exporter` runs as a sidecar or separate DaemonSet, scraping metrics from the local hostengine.

### What it measures

DCGM exposes **300+ metrics** organized into **field groups**. Key categories:

- **Profiling metrics** (field group 1000+): SM occupancy, Tensor Core utilization, DRAM bandwidth, PCIe/NVLink bandwidth, NVLink TX/RX throughput per link
- **Utilization and clocks** (field group 100): SM active %, memory active %, graphics/SM/memory clocks
- **Memory** (field group 110): Free, used, reserved memory; memory bandwidth utilization
- **Power and temperature** (field group 120): Power usage, power limit, thermal violations, throttle reasons
- **ECC and reliability** (field group 130): Single-bit ECC errors, double-bit ECC errors, PCIe replay count, Xid errors
- **NVLink** (field group 150): Per-link throughput, errors, and recovery counts
- **Processes and accounting**: Per-process GPU usage, job-level energy consumption (when accounting mode enabled)

The full list of fields and their IDs is documented in the DCGM API reference: [DCGM Field Identifiers](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html)

### When to use it

- **Production monitoring:** Deploy DCGM + dcgm-exporter + Prometheus to monitor a fleet of GPU nodes continuously
- **Health checks:** Run `dcgmi diag` to validate GPU hardware before deploying workloads (e.g., during node provisioning or after maintenance)
- **Observability at scale:** DCGM is the only NVIDIA tool designed for 100s–1000s of GPUs (DGX SuperPOD, GKE clusters)
- **Advanced profiling metrics:** Need Tensor Core utilization, NVLink per-link bandwidth, or SM pipeline stalls? These are exposed by DCGM but not by nvidia-smi.
- **Integration with Kubernetes or Slurm:** DCGM has native support for job-level tagging and accounting, allowing you to attribute GPU usage to specific jobs or users.

Do NOT use DCGM for:
- Interactive, one-off queries (use nvidia-smi)
- Kernel-level profiling (use `nsys` or `ncu`)
- Host-side or network-level metrics (use Prometheus node_exporter, `ethtool`, `perftest`)

### Commands (dcgmi)

#### Start the hostengine

On most systems, `nv-hostengine` is started automatically by a systemd service. To start manually (e.g., in a container):

```bash
nv-hostengine
```

Or run in standalone mode (non-daemon, foreground):

```bash
nv-hostengine --no-daemon
```

#### Discover GPUs

```bash
dcgmi discovery -l
```

Lists all GPUs visible to DCGM with their index, GPU ID (DCGM's internal handle), UUID, and device name. This is the DCGM equivalent of `nvidia-smi -L`.

Example output schema:
```
8 GPUs found.
+--------+------------------------------------------------------+--------+
| GPU ID | Device Information                                   | Status |
+========+======================================================+========+
| 0      | Name: NVIDIA H100 80GB HBM3                          | Ok     |
|        | UUID: GPU-...                                        |        |
+--------+------------------------------------------------------+--------+
...
```

#### Monitor metrics (streaming table)

```bash
dcgmi dmon
```

Continuously samples and prints a table of key GPU metrics at 1-second intervals (similar to `nvidia-smi dmon`, but with more fields and lower overhead when monitoring many GPUs). Default fields include: SM active %, memory utilization %, SM clock, memory clock, power, temperature, and PCIe/NVLink RX/TX throughput.

**Customizing fields:**

```bash
dcgmi dmon -e 100,150,155,204,1001,1002
```

- `-e <field_ids>`: Comma-separated list of DCGM field IDs to display (see [field ID reference](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html))
- `-c <count>`: Number of iterations (default: infinite)
- `-d <delay_ms>`: Sampling interval in milliseconds (default: 1000)

Example field IDs:
- `100`: SM active %
- `150`: Memory active %
- `155`: Memory bandwidth utilization %
- `204`: Power usage (W)
- `1001`: Tensor Core active %
- `1002`: DRAM active %
- `1005`: NVLink TX throughput (bytes/s)

#### Run diagnostics

```bash
dcgmi diag -r 1
```

Runs a quick hardware validation test (level 1: deployment test). Checks for:
- GPU presence and driver communication
- Memory integrity (quick scan)
- PCIe link width and speed
- Basic compute sanity checks

**Diagnostic levels:**
- `-r 1`: Deployment test (~1 min) — validates hardware is present and functional
- `-r 2`: Medium stress test (~10 min) — memory stress, compute stress, PCIe bandwidth
- `-r 3`: Long stress test (~30+ min) — extended burn-in, ECC scrub, thermal validation

Use level 1 for CI/CD pre-deployment checks. Use level 2 or 3 for hardware bringup, post-maintenance validation, or debugging flaky GPUs.

#### Field groups

DCGM organizes metrics into **field groups** for convenience. You can query a predefined group or define custom groups.

List available field groups:
```bash
dcgmi group -l
```

Create a custom field group:
```bash
dcgmi group -c my_group -a 100,150,204
```

Watch a field group:
```bash
dcgmi dmon -g 1  # Use field group 1 (default)
```

#### Health checks

```bash
dcgmi health -c
```

Runs continuous health monitoring, flagging issues like:
- Thermal violations
- Power limit violations
- ECC errors exceeding thresholds
- PCIe replay errors
- Xid errors

You can configure custom health policies (thresholds and responses) via DCGM's policy API.

### dcgm-exporter: Prometheus integration

**dcgm-exporter** is a Prometheus exporter that scrapes DCGM metrics and exposes them in Prometheus format on an HTTP endpoint (default: `:9400/metrics`). It's the standard way to feed GPU telemetry into Prometheus-based observability stacks.

#### Deployment on Kubernetes

Typical pattern (used in GKE, DGX Cloud, DGX SuperPOD):

1. Deploy `nv-hostengine` as a **DaemonSet** (one pod per GPU node)
2. Deploy `dcgm-exporter` as a **sidecar container** in the same pod, or as a separate DaemonSet
3. Configure Prometheus to scrape `:9400/metrics` from each GPU node's dcgm-exporter
4. Visualize in Grafana with NVIDIA's reference dashboards or custom queries

Example manifests and setup instructions are covered in **lab-10** (Observability and Fleet-Scale Debugging).

#### Exported metrics

dcgm-exporter exposes **100+ Prometheus metrics** with labels for GPU UUID, GPU index, and (optionally) Kubernetes pod/container/namespace. Key metrics:

- `DCGM_FI_DEV_GPU_UTIL`: GPU utilization % (SM active)
- `DCGM_FI_DEV_MEM_COPY_UTIL`: Memory utilization %
- `DCGM_FI_DEV_POWER_USAGE`: Power usage (W)
- `DCGM_FI_DEV_GPU_TEMP`: GPU temperature (°C)
- `DCGM_FI_DEV_FB_USED`: Frame buffer (HBM) used (MB)
- `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL`: Total NVLink bandwidth (bytes/s)
- `DCGM_FI_DEV_PCIE_RX_THROUGHPUT` / `TX_THROUGHPUT`: PCIe throughput (bytes/s)
- `DCGM_FI_DEV_XID_ERRORS`: Xid error count (by Xid type)

Full list: [dcgm-exporter metrics reference](https://github.com/NVIDIA/dcgm-exporter#metrics)

**Forward reference:** Lab-10 demonstrates deploying dcgm-exporter on the GKE A3 cluster, scraping metrics into Prometheus, and using them to debug a stalled NCCL collective.

### DCGM vs. nvidia-smi

| Aspect | nvidia-smi | DCGM |
| :--- | :--- | :--- |
| **Use case** | Interactive CLI inspection | Production monitoring, diagnostics, fleet observability |
| **Architecture** | Thin wrapper, calls NVML directly per invocation | Agent-based, persistent daemon, batched sampling |
| **Overhead** | High (100–500 ms per call) | Low (<10 ms for batch queries) |
| **Metrics** | ~50 metrics (core inventory + telemetry) | 300+ metrics (includes profiling, NVLink per-link, Tensor Core util) |
| **Prometheus integration** | None (manual scripting required) | Native (`dcgm-exporter`) |
| **Diagnostics** | None (telemetry only) | `dcgmi diag` (stress tests, validation) |
| **Multi-node** | Per-host only | Supported (DCGM can aggregate across nodes via DCGM Host Engine) |
| **When to use** | Quick checks, interactive debugging | Continuous monitoring, health checks, fleet dashboards |

---

## nvtop and gpustat: Community tools

In addition to NVIDIA's official tools, two popular community-maintained utilities provide alternative UIs for GPU monitoring:

### nvtop

**nvtop** is an `htop`-like GPU monitoring tool with a real-time TUI (text UI) that displays:
- GPU utilization, memory usage, temperature, power, and clocks (per GPU)
- Process list with per-process GPU memory usage
- Historical sparkline graphs for each metric
- Support for multiple GPUs in a scrollable view

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install nvtop

# From source (if packaged version outdated)
git clone https://github.com/Syllo/nvtop.git
cd nvtop
mkdir build && cd build
cmake ..
make
sudo make install
```

**Usage:**
```bash
nvtop
```

**When to use it:** Interactive monitoring during development or debugging. Easier to read at a glance than `nvidia-smi`, especially when monitoring multiple GPUs or processes over time. Not suitable for scripting or fleet monitoring (use DCGM for that).

### gpustat

**gpustat** is a Python-based command-line tool that provides a clean, colorized, compact summary of GPU status. It's essentially a prettier, more concise version of `nvidia-smi`.

**Installation:**
```bash
pip install gpustat
```

**Usage:**
```bash
gpustat        # One-time summary
gpustat -i 1   # Continuous monitoring (1-second interval)
```

**Output** (example schema):
```
[0] Tesla H100 80GB | 45°C, 75% | 65535 / 81920 MB | user1(12345):32768MB user2(67890):32767MB
[1] Tesla H100 80GB | 42°C, 80% | 70000 / 81920 MB | user3(11111):35000MB
```

**When to use it:** Same as nvtop — quick, human-friendly monitoring during interactive work. Can be easily embedded in shell prompts or tmux status bars.

---

## Portability note

All tools in this doc (**nvidia-smi**, **NVML**, **DCGM**, **nvtop**, **gpustat**) are **platform-agnostic** and work identically on:
- Bare-metal NVIDIA GPU hosts (DGX, SuperMicro, generic servers)
- VMs with GPU passthrough (GCP, AWS, Azure)
- Kubernetes pods with GPU access (GKE, EKS, AKS, on-prem)
- Docker containers with `--gpus` flag or Kubernetes device plugin

### On GKE with Container-Optimized OS (COS)

On GKE A3/A4 clusters, the GPU nodes run **Container-Optimized OS**, a minimal, read-only OS with the NVIDIA driver pre-installed but no shell access to the host. This means:

- **You CANNOT ssh into the node and run nvidia-smi directly on the host.**
- Instead, you run nvidia-smi and dcgmi **inside a GPU-enabled pod** (a pod with `nvidia.com/gpu` resource request). The pod sees the GPUs via the device plugin, and nvidia-smi works normally.

Example: To run nvidia-smi on a GKE GPU node:

```bash
kubectl run gpu-test --image=nvidia/cuda:12.3.1-base-ubuntu22.04 \
  --limits=nvidia.com/gpu=1 --rm -it -- nvidia-smi
```

The nvidia-smi command runs inside the container but queries the host's GPUs via the driver mounted into the container by the GKE GPU device plugin.

**DCGM deployment:** On GKE (and most Kubernetes GPU clusters), `nv-hostengine` is deployed as a **DaemonSet** (privileged pod with host access) so it can see all GPUs on the node. Application pods query metrics via the DCGM API exposed by the hostengine pod, or metrics are scraped by `dcgm-exporter` and sent to Prometheus.

This pattern (GPU tools running inside containers with driver/device access) is standard across all Kubernetes GPU deployments (NVIDIA GPU Operator, GKE, EKS, AKS, on-prem).

---

## Tools in this layer feed into

- **Labs:** Lab-01 uses nvidia-smi to inspect GPU architecture properties; Lab-10 deploys DCGM + dcgm-exporter + Prometheus for fleet observability
- **Diagnostics:** nvidia-smi and DCGM output are inputs to health checks and Xid error interpretation (covered in Part I, section 2: Driver/CUDA troubleshooting)
- **Profiling:** nvidia-smi topology (`topo -m`) informs GPU placement decisions for multi-GPU training (Part I, section 4: NVLink/HGX)
- **Observability pipelines:** dcgm-exporter is the foundation of GPU telemetry for Prometheus/Grafana dashboards (Part III, section 10)
- **Reference docs:** The glossary (`../../reference/glossary.md`) defines terms like ECC, HBM, NVLink, Xid; the tool cheatsheet (`../../reference/tool-cheatsheets.md`) provides quick command lookups

---

## Practice

- **Lab-01 (GPU Architecture Inspection):** Uses `nvidia-smi -q`, `nvidia-smi topo -m`, and pynvml to capture GPU inventory, memory hierarchy, and NVLink topology on the live GKE A3 H100 cluster
- **Lab-10 (Observability and Fleet-Scale Debugging):** Deploys DCGM + dcgm-exporter as a DaemonSet, configures Prometheus to scrape GPU metrics, and uses DCGM telemetry to diagnose a stalled NCCL all-reduce

Real command outputs and interpretation workflows for all tools in this doc are captured in those labs.

---

## Further reading

- NVIDIA NVML API Reference: [https://docs.nvidia.com/deploy/nvml-api/](https://docs.nvidia.com/deploy/nvml-api/)
- DCGM Documentation: [https://docs.nvidia.com/datacenter/dcgm/](https://docs.nvidia.com/datacenter/dcgm/)
- DCGM Field Identifiers: [https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html)
- dcgm-exporter GitHub: [https://github.com/NVIDIA/dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter)
- nvtop GitHub: [https://github.com/Syllo/nvtop](https://github.com/Syllo/nvtop)
- gpustat GitHub: [https://github.com/wookayin/gpustat](https://github.com/wookayin/gpustat)
