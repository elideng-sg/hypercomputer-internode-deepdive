# Lab-Build Gotchas — real bugs hit while building these labs on live A3

Every lab in this guide was built and captured against **live** A3 clusters (us-central1 2-node, asia-east1-c 3-node), not a simulator. That process surfaced a series of real, non-obvious failures — in the NVIDIA toolchain, in NCCL/`torch.distributed`, in GKE's managed environment, and in the capture harness itself. **These are exactly the traps a reader will hit doing this work for real**, so they are collected here as a first-class reference rather than buried in individual labs.

Each entry: the **symptom** as it actually appeared, the **root cause**, the **fix**, and **where** it bit us. Signatures are copied from real runs (honesty rule: nothing here is invented). This file is appended to as new labs are built.

> **How to read this alongside the labs.** Each lab README calls out its own headline gotcha inline; this file is the *cross-lab index* so a reader debugging (say) a `set -e` abort or an `ncu` permission error can find every occurrence in one place. See also [tool-cheatsheets.md](tool-cheatsheets.md) (run-vs-named honesty notes) and [doc-16](../docs/part5-operations-diagnostics/16-diagnostic-method.md) (the triage method these lessons instantiate).

---

## Harness / capture (`scripts/lib_capture.sh`)

### G1 — `set -euo pipefail` is inherited, and a helper ending in a false test aborts the whole script

**Symptom (lab-15, lab-13).** A run script sourced `lib_capture.sh`, backgrounded per-node launches inside a helper function, and the *whole script exited* mid-phase — the `EXIT` trap fired, tore down the workbench pods, and restored the `gpu-holder` before any fault was captured. No error message; just an early, clean-looking exit.

**Root cause.** `scripts/lib_capture.sh` sets `set -euo pipefail` at line 3. Any script that `source`s it inherits `errexit`. The launch helper's *last* statement was a bare short-circuit test:

```bash
launch_phase() {
  ...
  for node_r in 0 1 2; do
    kubectl exec "$pod" ... &
    [ "$node_r" -eq 0 ] && sleep 6   # <-- returns 1 when node_r != 0
  done
}   # function return code = last command = 1  →  errexit kills the script
```

For `node_r != 0` the `[ ... ]` test is false, so the loop's — and therefore the function's — return code is `1`. Under `errexit`, calling that function as a bare statement aborts the script.

**Fix.** End such helpers with an explicit `return 0`:

```bash
    [ "$node_r" -eq 0 ] && sleep 6
  done
  return 0   # never let the trailing false test trip the inherited `set -e`
}
```

**Lesson.** A backgrounded-launch helper must not let its exit status be the value of a conditional. Either end with `return 0`, use `|| true`, or write `if [ ... ]; then sleep 6; fi` (whose status is 0 when not taken). This bites *any* script that sources a `set -e` library — a very common shape.

---

## NCCL / torch.distributed

### G2 — a straggler before the *first* collective stalls comm **init**, not the PG-work timeout

**Symptom (lab-15).** To demonstrate the classic "one rank is late → the group hangs → watchdog aborts at the PG timeout" signature, the first attempt simply had rank 16 `time.sleep(70)` before the first `all_reduce` with `PG_TIMEOUT=45`. Expectation: survivors abort at ~45 s. **What actually happened: all 23 other ranks blocked for the full 70 s and then the job *completed normally* — no watchdog abort at all.**

**Root cause.** NCCL builds the communicator **lazily on the first collective** (`ncclCommInitRank`). A rank that is late to the *very first* collective stalls comm **initialization**, which is gated by NCCL's own long **bootstrap** timeout — a different (much longer) clock than the per-work `Timeout(ms)` the watchdog enforces on an *established* collective. So the group just waited out the straggler's sleep and proceeded.

**Fix.** Run a **warmup all-reduce with all ranks present** to build the communicator *first*, and only then let the straggler sleep before a subsequent collective:

```python
dist.all_reduce(buf); torch.cuda.synchronize(); dist.barrier()   # comm now established
if rank == straggler_rank and straggler_sleep > 0:
    time.sleep(straggler_sleep)          # now stalls an ESTABLISHED collective
for it in range(iters):
    dist.all_reduce(buf); torch.cuda.synchronize()
```

Now the survivors abort at exactly `Timeout(ms)=45000` (`ran for 45008 milliseconds` observed). **Lesson: init hangs and collective hangs run on different clocks** — this is itself documented as a teaching point in [lab-15](../labs/lab-15-internode-comms-debug/) and [doc-18](../docs/part5-operations-diagnostics/18-internode-comms-troubleshooting.md).

### G3 — `hostNetwork: true` makes `hostname` return the **node** name, breaking per-rank identity

**Symptom (lab-13a JobSet gang).** Workbench pods run with `hostNetwork: true` (so NCCL binds the node's `eth0` for inter-node traffic). Code that derived a pod/rank identity from `hostname` got the **node** name for every pod, and a per-rank regex that end-anchored on a digit (`worker-<n>$`) failed because JobSet pod names carry a **random suffix** (`worker-<replica>-<completion>-<rand>`).

**Root cause.** `hostNetwork: true` puts the pod in the host's UTS namespace, so `hostname` == node hostname, not pod name. And JobSet/Job pod names are not cleanly end-anchored.

**Fix.** Take the node-rank from the JobSet **`jobset.sigs.k8s.io/job-index`** annotation via the downward API, not from `hostname`/pod-name parsing. (For the manual c10d labs, pass `NODE_RANK` explicitly on the launch command instead of inferring it.)

### G4 — no `sshd` in the NGC image → can't `mpirun` `all_reduce_perf` across nodes

**Symptom (lab-06).** The canonical multi-node `nccl-tests` launch (`mpirun -H node1,node2 all_reduce_perf …`) can't be used: `nvcr.io/nvidia/pytorch:24.10-py3` ships no `sshd`, so MPI has no way to start remote ranks.

**Fix.** Launch a `torch.distributed` all-reduce (`allreduce_bench.py`) via manual `RANK`/`WORLD_SIZE`/`MASTER_ADDR` c10d env instead — it computes the **same** `busbw = algbw·2(n−1)/n`. The single-node 8-GPU sweep still uses the `all_reduce_perf` binary (no remote launch needed). See [tool-cheatsheets.md](tool-cheatsheets.md) → nccl-tests.

---

## NVIDIA toolchain on managed GKE (Container-Optimized OS)

### G5 — `ncu` fails with `ERR_NVGPUCTRPERM` (the profiling-counter permission wall)

**Symptom (lab-03).** `ncu --set full -o out python3 gemm.py` connects to the process, then:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

**Root cause.** GKE COS loads the NVIDIA kernel module with `NVreg_RestrictProfilingToAdminUsers=1`. Reading hardware performance counters (what `ncu` needs) then requires **`CAP_SYS_ADMIN`** — an ordinary (even root-in-container) process without that capability is refused. `nsys` is unaffected because it uses CUPTI *activity tracing*, not the perf counters.

**Fix.** Run the profiled pod with `securityContext.capabilities.add: ["SYS_ADMIN"]` (or `privileged: true`) and add `--target-processes all`. This is the "half of profiling lab-03 couldn't finish" that **[lab-14](../labs/lab-14-single-gpu-health-triage/)** completes. Note the scope trap: `ncu --set full` profiles *every* kernel with the full metric set — on `gemm.py`'s large GEMMs that is enormously slow, so bound it with `--launch-count N --launch-skip M` (profile a handful of kernels), or it looks like a hang.

### G6 — `dmesg` (and therefore `dmesg | grep Xid`) is blocked in an unprivileged container

**Symptom (lab-02).** The textbook XID-triage step `dmesg | grep -i -E "NVRM|Xid"` returns nothing usable from inside a normal pod (`dmesg: read kernel buffer failed: Operation not permitted`), and `kubectl debug node/...` for host `dmesg` is also restricted on COS.

**Root cause.** Reading the kernel ring buffer needs `CAP_SYSLOG`/host access, which unprivileged GKE pods don't have. **This restriction is itself the lesson.**

**Fix / the GKE-native path.** XIDs surface *without* `dmesg`:
- **Node Problem Detector → Cloud Logging** — query the node's logs for `NVRM: Xid`; NPD also raises a node **condition**/event (`kubectl get events`).
- **`DCGM_FI_DEV_XID_ERRORS`** — scraped by the managed `dcgm-exporter` (the lab-10 fleet pipeline), queryable in Cloud Monitoring / via `curl :9400/metrics`.

A **privileged** pod *can* read `dmesg` (used in lab-14 to show the contrast), but you should never depend on that in production tooling — use the two managed paths above. See [xid-table.md](xid-table.md).

### G7 — `dcgmi` is **not** in the PyTorch NGC image

**Symptom (lab-02).** `dcgmi diag -r 2` → `dcgmi: command not found` inside `nvcr.io/nvidia/pytorch:24.10-py3`.

**Fix.** Either `apt-get install -y datacenter-gpu-manager` into the running container, or (cleaner) run diagnostics as a dedicated **Job** using the DCGM image `nvcr.io/nvidia/cloud-native/dcgm:3.3.8-1-ubuntu22.04` with `command: ["dcgmi","diag","-r","2"]`. **Caveat:** the Job form needs a **free GPU** — if a borrow-window workbench already holds all 8 GPUs on the node, there is no GPU for a second pod, so run `dcgmi` *inside* the workbench (install it there) rather than as a separate Job.

### G8 — `nvbandwidth` / `bandwidthTest` binaries are **absent** from the NGC image

**Symptom (lab-03).** `nvbandwidth …` and the CUDA-samples `bandwidthTest` both → `command not found`.

**Fix.** Get HBM/interconnect bandwidth from what *is* present: `nccl-tests` all-reduce (inter-GPU) and the intra-node NVLink mesh read (lab-04). `deviceQuery` *is* present but produced a famously bad global-memory reading (G9).

### G9 — `deviceQuery` reported a bad global-memory value

**Symptom (lab-01).** `deviceQuery` printed an implausible global-memory figure — a caught **bad reading**, documented in the [lab-01 README](../labs/lab-01-gpu-arch-inspect/README.md) rather than trusted.

**Lesson.** Cross-check any single tool's number against a second source (`nvidia-smi -q -d MEMORY`, the NGC spec) before quoting it — the guide's *measured-not-asserted* discipline in miniature.

---

### G10 — `nvidia-smi -q -d ROW_REMAP` / `PCIE` → "Failed to parse --display/-d flags"

**Symptom (lab-14, Phase A).** The idle health baseline read

```
nvidia-smi -i 0 -q -d ECC,PERFORMANCE,CLOCK,POWER,TEMPERATURE,ROW_REMAP,PCIE
```

aborted immediately with `Failed to parse --display/-d flags`, and — because the lab sources the `set -e` harness (G1) — took the whole Phase A capture with it before anything was written.

**Root cause.** `ROW_REMAP` and `PCIE` are **not** valid `-d`/`--display` group tokens. The real group name for remapped-row health is **`ROW_REMAPPER`**, and there is no `PCIE` group at all (PCIe replay/link info comes from the default `-q` output or `nvidia-smi -q -d SUPPORTED_CLOCKS`-adjacent sections, not a `-d PCIE` group). One bad token rejects the *entire* comma list, so a single typo silently kills a multi-group query.

**Fix.** Use the exact group names `nvidia-smi --help-query-gpu`-style docs list: `ECC,PERFORMANCE,CLOCK,POWER,TEMPERATURE,ROW_REMAPPER`. Drop `PCIE`. Validated groups used in lab-14's idle read now parse cleanly (`Remapped Rows`, `Row Remapper` histogram present in `health_idle_full.txt`).

---

## DCGM version-matching on a CUDA-13 (R580) driver

The asia-east1-c pool runs driver **580.126.20** (Open Kernel Module, CUDA-13 era) — newer than us-central1's 535.x. That exposed a cluster of DCGM problems that do **not** appear on the older driver, and which together mean **no stock public DCGM image can run `dcgmi diag` (`-r 3`) on this node today.** lab-02's `-r 2` remains the working reference because it ran against the older us-central1 driver.

### G11 — the **managed** `dcgm-exporter` does **not** export an XID field by default

**Symptom (lab-14, Phase F).** Scraping the GKE-managed exporter (`kubectl port-forward -n gke-managed-system pod/<dcgm-exporter> 9400:9400`; `curl :9400/metrics`) for `DCGM_FI_DEV_XID_ERRORS` returned **nothing** — the field is simply absent from the managed exporter's default field set.

**Root cause.** GKE's managed `dcgm-exporter` ships a curated `DCGM_FI_*` field set (FB used/free, temperature, power, SM/mem/graphics clocks, GPU/mem utilization, and the PROF_* pipe/tensor/DRAM-active metrics). `DCGM_FI_DEV_XID_ERRORS` is **not** in it. So the tidy "just query `DCGM_FI_DEV_XID_ERRORS`" story in a lot of docs does not hold on managed GKE out of the box.

**Fix / correction.** On managed GKE, XIDs surface through **Node Problem Detector → Cloud Logging** (`NVRM: Xid` log lines) and the NPD-raised node **condition/event** — *not* the managed exporter's metrics. To get the XID metric you would have to run **your own** dcgm-exporter with a custom `csv` including `DCGM_FI_DEV_XID_ERRORS`. The captured field list is in `assets/lab-14/dcgm_xid_metric.txt` (19 fields, XID not among them). This corrects earlier wording in [xid-table.md](xid-table.md) and [tool-cheatsheets.md](tool-cheatsheets.md).

### G12 — DCGM **3.3.8** image: "Detected unsupported Cuda version"

**Symptom (lab-14, Phase D).** Running `dcgmi diag` from `nvcr.io/nvidia/cloud-native/dcgm:3.3.8-1-ubuntu22.04` (the image G7 recommends, which works on us-central1) failed on asia-east1-c with **"Detected unsupported Cuda version"** — the 3.3.x diagnostic plugins predate CUDA-13 and refuse to run against the R580 driver.

**Fix (partial).** Move to a DCGM 4.4.x image that matches the driver era (the managed exporter here is 4.4.1). That clears the "unsupported Cuda version" error — but then hits G13.

### G13 — DCGM **4.4.0** image: missing `plugins/cuda13/` → no diag plugins

**Symptom (lab-14, Phase D).** `nvcr.io/nvidia/cloud-native/dcgm:4.4.0-1-ubuntu22.04` gets past the version check but `dcgmi diag -r 3` then fails:

```
Error: Unable to complete diagnostic for entities *,cpu:*. Return: (-30) : DCGM GPU Diagnostic returned an error
Error: Cannot load plugins. Unable to change to the plugin dir
'/usr/libexec/datacenter-gpu-manager-4/plugins/cuda13/': 'No such file or directory'
```

**Root cause.** The 4.4.0 image ships diagnostic plugins for older CUDA ABIs but **not** a `cuda13/` plugin directory, so on a CUDA-13 driver there are no loadable diag plugins. Probing `ubuntu24.04`, `4.3.1`, `4.4.0-2` tags all `ImagePullBackOff` (they don't exist on NGC).

**Fix / honest state.** **Unresolved by available public images** as of this capture: there is no stock DCGM diag image with cuda13 plugins for R580. `dcgmi discovery`/health-watch and the lighter checks still work; the deep `diag -r N` suite does not. lab-14 therefore captures this as a **documented version-matching gotcha** (`assets/lab-14/dcgm_diag_r3.txt`) rather than a passing run — and lab-02's `-r 2` on the older driver stays the reference for what a green diag looks like. **Lesson: pin your DCGM image to the node's driver/CUDA generation, and verify `dcgmi diag` actually runs there before you write it into a runbook** — the diag plugins, not just the daemon, must match the CUDA ABI.

---

*(Appended as labs are built. Next: lab-16/lab-17 cluster & day-2 triage — see below once captured.)*
