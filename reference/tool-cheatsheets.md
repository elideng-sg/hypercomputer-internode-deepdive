# NVIDIA Tool Command Cheat Sheets

Quick command invocations for the key tools used throughout the labs. Each section lists the commands **actually run on the live A3 clusters** (with the lab that runs them), so this file doubles as an index of *where* each tool is exercised. Detailed interpretation of each tool's output lives in the `docs/toolkit/` deep-dives (T1–T6).

> **Honesty note — run vs. named.** Two networking tools in the toolkit, **`perftest`** (`ib_write_bw`…) and **`ethtool`**, are **never actually invoked** in any lab on these clusters: A3 High is gVNIC / GPUDirect-TCPX with **no InfiniBand/RoCE fabric**, so their RDMA/NIC-counter use cases don't apply here. They are documented in [`docs/toolkit/T5-networking-fabric-tools.md`](../docs/toolkit/T5-networking-fabric-tools.md) as reference and are flagged below as *named-but-not-run*. Likewise `ncu` runs but **fails by design** on managed GKE (privilege), and `nvbandwidth`'s binary is **absent** from the NGC image — both are honest, captured negative results, not omissions.
>
> Most GPU commands are executed inside a pod via `kubectl exec <pod> -- <cmd>`; run.sh scripts wrap them through `cap_run` (`scripts/lib_capture.sh`), and every captured `assets/*/*.txt` starts with a `# cmd:` provenance header echoing the exact command.

---

## nvidia-smi

**Purpose:** GPU monitoring and inventory — query properties, utilization, temperature, clocks, topology, NVLink, and running processes. *(Run in labs 01, 02, 04, 11.)*

```bash
kubectl exec gpu-debug -- nvidia-smi                      # inventory: GPUs, driver, mem, procs (lab-01)
kubectl exec gpu-debug -- nvidia-smi -q                   # full per-GPU dump (lab-01 smi-q.txt)
kubectl exec gpu-debug -- nvidia-smi topo -m              # GPU/NIC topology matrix (lab-01, lab-04 8-GPU, lab-11)
kubectl exec gpu-debug -- nvidia-smi -q -d ECC,PERFORMANCE,TEMPERATURE,POWER   # health groups (lab-02)
kubectl exec gpu-debug -- nvidia-smi --query-gpu=driver_version,name,vbios_version --format=csv   # scriptable inventory (lab-02)
kubectl exec "$POD" -- nvidia-smi nvlink --status -i 0    # per-GPU NVLink link state (lab-04, lab-11)
```

- Useful `-q -d <group>`s: `ECC`, `PERFORMANCE`, `TEMPERATURE`, `POWER` (run in lab-02); `ROW_REMAP`, `PAGE_RETIREMENT`, `PCIE` are the health-triage groups referenced by [xid-table.md](xid-table.md) (produced live in **lab-14**).
- `nvidia-smi dmon` / `pmon` (streaming per-GPU/per-process sampling) are **not yet run** in any existing lab — introduced in **lab-14** (throttle under load) and **lab-17** (day-2 monitoring).

## DCGM (dcgmi)

**Purpose:** Datacenter GPU management — health checks, stress/diagnostics, telemetry, field queries. *(Run in lab 02; the fleet pipeline in lab 10.)*

```bash
kubectl exec gpu-debug -- dcgmi diag -r 2                 # medium diagnostic suite (lab-02 dcgm-diag.txt)
# or as a dedicated Job container:
#   command: ["dcgmi","diag","-r","2"]                    # (lab-02 manifest form)
```

- Diagnostic levels: `-r 1` (quick), `-r 2` (medium — the one run in lab-02), `-r 3` (long, deployment-grade — introduced in **lab-14**).
- **Lab-10's "DCGM" is the managed `dcgm-exporter` Prometheus pipeline, not the `dcgmi` CLI.** Metrics are scraped over HTTP, not via `dcgmi dmon`:
  ```bash
  kubectl port-forward -n gke-managed-system pod/dcgm-exporter-<id> 9400 &
  curl -s localhost:9400/metrics                          # (lab-10 dcgm-metrics-raw.txt)
  ```
  Key field IDs present in the scrape: `DCGM_FI_DEV_{SM_CLOCK,GPU_TEMP,POWER_USAGE,GPU_UTIL,FB_USED}` and the profiling fields `DCGM_FI_PROF_{GR_ENGINE_ACTIVE,SM_ACTIVE,PIPE_TENSOR_ACTIVE,DRAM_ACTIVE,PCIE_TX_BYTES,NVLINK_TX_BYTES}`. These drive the **lab-17** Grafana/PromQL dashboards.
- `dcgmi dmon -e <fields>` / `dcgmi health` — introduced in **lab-14** (single-GPU health) / **lab-17** (day-2); not run in the current labs.

## Nsight Systems (nsys)

**Purpose:** System-wide timeline profiling — CUDA kernels, CUDA API, NVTX ranges, CPU activity. Uses CUPTI activity tracing, so it needs **no elevated privilege** (unlike `ncu`). *(Run in lab 03.)*

```bash
kubectl exec gpu-debug -- nsys profile -t cuda,nvtx -o /work/gemm python3 /work/gemm.py   # capture (lab-03)
kubectl exec gpu-debug -- nsys stats /work/gemm.nsys-rep                                   # summarize the .nsys-rep (lab-03 nsys-stats.txt)
```

- Flags actually used: `-t cuda,nvtx` (trace domains), `-o <out>`. The multi-rank comm/compute-overlap analysis (HTA on an nsys/torch trace) is a **lab-17** exercise.

## Nsight Compute (ncu)

**Purpose:** Kernel-level profiling — per-kernel memory throughput, compute utilization, roofline. *(Attempted in lab 03 — **fails by design** on managed GKE.)*

```bash
kubectl exec gpu-debug -- ncu --set full -o /work/gemm_ncu python3 /work/gemm.py   # lab-03: ERRORS on GKE COS
```

- **Why it fails here (the lesson):** GKE Container-Optimized OS sets `NVreg_RestrictProfilingToAdminUsers=1`, and `ncu` needs the hardware performance counters, which require **`CAP_SYS_ADMIN` / privileged mode**. Captured verbatim in `assets/lab-03/ncu-output.txt` ("expected on GKE COS without CAP_SYS_ADMIN"). The privileged rerun (adding `securityContext` + `--target-processes all`) is the half of profiling that **lab-14** finally completes.

## nvbandwidth

**Purpose:** GPU memory / interconnect bandwidth — HBM, P2P over NVLink/PCIe, host↔device transfers. *(Attempted in lab 03 — **binary absent** from the NGC image; `bandwidthTest`/`deviceQuery` fallbacks.)*

```bash
nvbandwidth -t testcase=1,testcase=2                      # H2D / D2H — binary not present in the container (lab-03)
bandwidthTest --htod --dtoh                               # CUDA-samples fallback — also absent from image (lab-03)
kubectl exec gpu-debug -- bash -lc 'deviceQuery || /usr/local/cuda/extras/demo_suite/deviceQuery'   # ran (lab-01)
```

- On this image the P2P/HBM bandwidth numbers come from **`nccl-tests` all-reduce** (below) and the intra-node NVLink mesh in lab-04, not from `nvbandwidth`. `deviceQuery` ran in lab-01 but produced the famous bad global-memory reading (see [lab-01 README](../labs/lab-01-gpu-arch-inspect/README.md) — a caught bad reading).

## nccl-tests

**Purpose:** NCCL collective benchmarking — bandwidth/latency of all-reduce, all-gather, reduce-scatter, broadcast. *(The binary is run in lab 04; multi-node labs use a torch equivalent — see note.)*

```bash
kubectl exec "$POD" -- bash -lc 'NCCL_DEBUG=WARN all_reduce_perf -b 8 -e 8G -f 2 -g 8'   # 8-GPU intra-node sweep (lab-04)
```

- Flags: `-b` begin size, `-e` end size, `-f` step factor (×2), `-g` GPUs per process, `-n` iters, `-w` warmup. Binaries ship in `nvcr.io/nvidia/pytorch:24.10-py3` (NCCL 2.22.3).
- **Why the multi-node labs don't use the binary:** the NGC image has no `sshd`, so a 2-node `mpirun` launch of `all_reduce_perf` isn't possible. Labs 06/12/13 instead run a `torch.distributed` all-reduce (`allreduce_bench.py`) computing the **same** `busbw = algbw·2(n−1)/n`, launched via manual `RANK`/`WORLD_SIZE` c10d env (see [nccl-tunables.md](nccl-tunables.md) and lab-06). **lab-15** uses `NCCL_DEBUG=INFO` on this path to localize a slow/hung collective.

## perftest (ib_write_bw, ib_read_bw)

**Purpose:** Raw RDMA/RoCE fabric bandwidth & latency, independent of GPU/NCCL.

> **Named-but-not-run on these clusters.** A3 High is **gVNIC / GPUDirect-TCPX with no InfiniBand/RoCE HCA**, so there is no fabric for `ib_*_bw` to test — no lab invokes it. It is documented for RDMA fabrics (A3 Ultra H200 RoCE, A4/A4X) in [T5](../docs/toolkit/T5-networking-fabric-tools.md). Reference form only:

```bash
# reference only (RDMA fabric — not runnable on A3 High):
ib_write_bw -d mlx5_0 -i 1 -F --report_gbits            # server
ib_write_bw -d mlx5_0 -i 1 -F --report_gbits <server-ip> # client
```

## ethtool

**Purpose:** Ethernet NIC statistics/config — link speed, MTU, offload, RoCE counters, drops.

> **Named-but-not-run on these clusters.** No lab invokes `ethtool`; lab-05 (network-path inspection) reads NIC inventory from `/sys/class/net` + `/proc/net/dev` and node annotations instead (the `networkstatic/iperf3` image ships no `iproute2`/`ethtool`). It is documented in [T5](../docs/toolkit/T5-networking-fabric-tools.md); the live `ethtool -S` drop/counter deltas are a **lab-15** target (inter-node comms debug). Reference form:

```bash
# reference — NIC identity and counters (introduced live in lab-15):
ethtool -i eth0                                          # driver (expect: gve), firmware, bus-info
ethtool -S eth0                                          # per-queue rx/tx, drops, errors
```

## kubectl (GPU-specific)

**Purpose:** Cluster/workload inspection — GPU device-plugin, allocatable GPUs, placement, DWS, Kueue/JobSet status, events. *(Run across labs 02, 05, 07, 08, 10, 13.)*

```bash
# node / GPU inventory & topology
kubectl get nodes -l cloud.google.com/gke-nodepool=$LAB_NODEPOOL -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'   # (lib_capture.sh)
kubectl describe node $NODE | sed -n '/Allocatable/,/System Info/p'      # nvidia.com/gpu: 8 allocatable (lab-05)
kubectl get node $NODE -o jsonpath='{.metadata.annotations}' | tr ',' '\n' | grep -iE 'gpu|nic|tcpx|rdma'   # fabric annotations (lab-05)

# device plugin & GPU DaemonSets
kubectl get ds -n kube-system -o wide | awk 'NR==1 || /nvidia-gpu-device-plugin/'   # (lab-07)
kubectl get ds -A -o wide | grep -iE 'tcpx|fastsocket|rdma|nccl|gpudirect'          # confirm no TCPX/RDMA DS (lab-05)
kubectl get ds -n gke-managed-system dcgm-exporter -o wide                          # fleet telemetry (lab-10)

# scheduling gate / events
kubectl get events --field-selector reason=FailedScheduling --sort-by=.lastTimestamp | tail -5   # Insufficient nvidia.com/gpu (lab-07)
kubectl describe pod $POD | sed -n '/Events:/,$p'                                    # per-pod scheduling events

# Kueue admission & JobSet gang
kubectl get workloads -n default -o wide                                            # QuotaReserved/Admitted (lab-08, lab-13a)
kubectl get clusterqueue gpu-cq -o jsonpath='{.status}' | python3 -m json.tool      # quota usage (lab-08)
kubectl get jobset nccl-jobset -o wide                                              # gang structure (lab-08)
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=nccl-gang -o wide                # 1 pod/node placement (lab-13a)
```

- The OOMKilled/exit-137, crashloop, and Kueue-inadmissible triage forms of these commands (`kubectl get pod -o jsonpath='{...exitCode}'`, `describe` reason fields, `Workload` `Pending` conditions) are exercised in **lab-16**.

---

**Related:** [nccl-tunables.md](nccl-tunables.md) · [xid-table.md](xid-table.md) · [driver-matrix.md](driver-matrix.md) · toolkit deep-dives [T1–T6](../docs/toolkit/) · Part V scenario labs 14–17.
