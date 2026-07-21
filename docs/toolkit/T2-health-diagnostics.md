# T2: GPU Health and Diagnostics

## Overview

This reference covers the NVIDIA toolchain for GPU health monitoring, fault detection, and root-cause analysis. The tools range from quick smoke-tests to comprehensive diagnostics that can take hours. All tools in this doc are **symptom-oriented**: the goal is to surface faults, identify the failing component (GPU, memory, NVLink, thermal system, power delivery), and determine whether the issue is transient (correctable) or fatal (requires RMA).

Every technique is framed as:
- **Symptom**: What you observe (crashed job, performance drop, kernel panic)
- **Cause**: What the diagnostic reveals (ECC errors, thermal throttle, XID event)
- **Action**: What to do next (reboot, replace GPU, tune workload)

This doc cross-references the **XID Error Reference Table** (`../../reference/xid-table.md`), which maps specific XID codes to meanings and actions. That table is populated as XIDs are encountered in the labs (primarily Lab 02 and Lab 10).

---

## DCGM Diagnostics

**DCGM** (Data Center GPU Manager) provides a hierarchy of diagnostic test suites via `dcgmi diag`. Each diagnostic level runs progressively deeper tests, trading runtime for coverage.

### Diagnostic Levels

#### Level 1: Quick Health Check
```bash
dcgmi diag -r 1
```

**What it runs:**
- Software environment checks (driver version, CUDA libraries, permissions)
- GPU enumeration and basic PCIe connectivity
- Firmware version verification
- Basic memory test (small allocation/deallocation)

**Runtime:** ~10-30 seconds per GPU

**What it catches:**
- Missing or mismatched drivers
- GPUs that have fallen off the bus (XID 79)
- Firmware mismatches across a node
- Catastrophic memory failures

**When to use:**
- Pre-deployment validation
- After driver updates or reboots
- Quick smoke-test before launching a multi-hour training job

**Typical output:**
The diagnostic prints per-GPU pass/fail status and a summary. A pass at level 1 means the GPU is enumerable and has a functioning driver stack.

---

#### Level 2: Medium Diagnostics
```bash
dcgmi diag -r 2
```

**What it runs:**
- All Level 1 checks
- Extended PCIe bandwidth test (host-to-device, device-to-host)
- GPU memory bandwidth test (using DCGM's internal stress kernels)
- Light compute stress test (FP32/FP64 operations)
- NVLink connectivity and bandwidth verification (if present)

**Runtime:** ~2-5 minutes per GPU

**What it catches:**
- PCIe bandwidth degradation (x16 link running at x8, Gen3 instead of Gen4)
- Memory performance anomalies (bandwidth significantly below spec)
- NVLink lane failures or disabled links
- Early signs of compute instability

**When to use:**
- Pre-production validation of a new node
- After detecting performance regressions
- Periodic health checks (weekly/monthly)

**Symptom → Cause → Action example:**
- **Symptom:** Job runs 20% slower than expected
- **Cause:** Level 2 diag shows PCIe running at x8 instead of x16
- **Action:** Reseat the GPU, check for PCIe link errors in `dmesg`, inspect physical connection

---

#### Level 3: Extended Stress and Burn-in
```bash
dcgmi diag -r 3
```

**What it runs:**
- All Level 1 and Level 2 checks
- Extended memory stress test (walking-1s, moving-inversions patterns across full VRAM)
- Compute stress test at high utilization (sustained load on SMs and Tensor Cores)
- Power and thermal monitoring under load
- Extended NVLink traffic test

**Runtime:** ~10-30 minutes per GPU (configurable via `--parameters`)

**What it catches:**
- Intermittent memory errors that only appear under sustained load
- Thermal throttling (see **Throttle Reasons** section below)
- ECC errors (correctable and uncorrectable)
- NVLink CRC errors or replay storms
- Power delivery issues (GPU unable to sustain rated TDP)

**When to use:**
- Initial node burn-in (before adding to production cluster)
- After replacing a GPU or updating firmware
- When investigating intermittent training crashes or silent data corruption

**Symptom → Cause → Action example:**
- **Symptom:** Occasional NaN loss spikes during training
- **Cause:** Level 3 diag reports uncorrectable ECC errors in HBM2e
- **Action:** Check ECC counters (`nvidia-smi -q -d ECC`), verify row remapping status, RMA GPU if uncorrectable errors persist

---

#### Level 4: Comprehensive Hardware Validation
```bash
dcgmi diag -r 4
```

**What it runs:**
- All Level 1-3 checks
- Exhaustive memory test (multiple passes with different patterns, can run for hours)
- Extended PCIe stress (sustained traffic to catch transient link errors)
- Maximum power/thermal stress
- Optional: targeted tests (e.g., graphics, video encode/decode)

**Runtime:** ~1-4 hours per GPU (or longer with custom parameters)

**What it catches:**
- Rare intermittent faults (e.g., single-bit errors that occur once per million operations)
- Marginal hardware (passes Level 3 but fails extended soak)
- Firmware bugs that only manifest under extreme stress

**When to use:**
- Pre-deployment validation for critical workloads (large-scale training, inference serving)
- After transporting hardware (shipping-induced faults)
- When a GPU has a history of unexplained crashes but passes lower-level diagnostics

**Note:** Level 4 is disruptive (takes the GPU offline for hours). Schedule appropriately.

---

### Interpreting DCGM Diagnostic Output

DCGM diagnostics produce structured output with per-test and per-GPU results. Key sections:

- **Test Summary:** Lists each test (e.g., "PCIe Bandwidth", "Memory") with Pass/Fail/Skip/Warning status
- **Per-GPU Details:** Shows which GPU failed which test, with error codes and messages
- **Warning vs. Fail:** Warnings indicate degraded performance or non-fatal issues; Fails indicate hardware faults or missing prerequisites

**Common failure patterns:**
- `"Memory test failed: Uncorrectable ECC error"` → Check ECC counters, likely RMA
- `"PCIe bandwidth below threshold"` → Check link width/speed with `nvidia-smi -q -d PCIE`
- `"NVLink test failed: CRC errors on link X"` → Check NVLink topology and Fabric Manager logs

---

## ECC, Page Retirement, and Row Remapping

**ECC (Error Correction Code)** memory protects GPU DRAM (HBM2e on H100) from single-bit errors. When the GPU detects errors, it may **retire pages** (mark them as unusable) or perform **row remapping** (remap faulty rows to spares).

### Querying ECC Status
```bash
nvidia-smi -q -d ECC
```

**Output includes:**
- **Volatile (current boot) ECC counters:** Single-bit (correctable) and double-bit (uncorrectable) errors since driver load
- **Aggregate (lifetime) ECC counters:** Total errors across all boots
- **ECC mode:** Enabled or Disabled (H100 ships with ECC enabled by default; do not disable for production AI workloads)

**Interpretation:**
- **Single-bit errors (correctable):** The ECC hardware corrected the error; data integrity is maintained. Occasional single-bit errors are normal (cosmic rays, DRAM aging). Monitor the rate.
- **Double-bit errors (uncorrectable):** The hardware could not correct the error; the kernel or application may have received corrupt data. **This is a fatal fault** and often triggers an XID event (e.g., XID 48, 63, 64, 94, 95).

**Symptom → Cause → Action:**
- **Symptom:** Training job crashes with "CUDA error: unspecified launch failure"
- **Cause:** `nvidia-smi -q -d ECC` shows uncorrectable errors
- **Action:** Check XID log (see below), verify error is not transient, RMA GPU if errors recur

---

### Page Retirement
```bash
nvidia-smi -q -d PAGE_RETIREMENT
```

**What it shows:**
- **Retired pages (current and pending):** Memory pages that have been marked unusable due to excessive ECC errors
- **Retirement reason:** Single-bit (correctable errors exceeded threshold) or double-bit (uncorrectable error)
- **Retirement limit:** GPUs have a finite number of spare pages (typically ~512-1024 for H100); once exhausted, the GPU may refuse to initialize

**Interpretation:**
- A few retired pages is normal over the GPU's lifetime (especially for long-running data-center GPUs)
- Rapid page retirement (many pages in days/weeks) indicates failing DRAM
- If the GPU reaches the retirement limit, it may enter a "retired page limit exceeded" state and refuse to load

**Action:**
- Monitor retirement rate: `watch -n 10 'nvidia-smi -q -d PAGE_RETIREMENT'`
- If retirement accelerates, plan for GPU replacement
- If GPU hits retirement limit, it is no longer usable (RMA required)

---

### Row Remapping
```bash
nvidia-smi -q -d ROW_REMAP
```

**What it shows:**
- **Remapping status:** Whether row remapping is available, pending, in-progress, or failed
- **Correctable/uncorrectable remap counts:** Number of rows remapped due to ECC errors

**Interpretation:**
- **Row remapping** is a GPU firmware feature that transparently remaps failing DRAM rows to spare rows (similar to bad-sector remapping on disks)
- When enabled, the GPU can tolerate localized DRAM defects without retiring large numbers of pages
- **Pending remap:** The GPU has queued a row remap; it will occur on the next GPU reset
- **Remap failed:** The GPU ran out of spare rows or encountered an error during remapping; this is a fatal condition

**Symptom → Cause → Action:**
- **Symptom:** Frequent ECC errors on the same memory region
- **Cause:** `nvidia-smi -q -d ROW_REMAP` shows high remap counts or pending remaps
- **Action:** Reboot the node to allow pending remaps to complete, then re-test with DCGM Level 3. If remaps continue to accumulate, RMA the GPU.

**Cross-reference:** Row remapping may trigger XID 63 ("Row Remapping") or XID 64 ("Row Remapping Failure"). See `../../reference/xid-table.md`.

---

## XID Errors

**XID** (eXception ID) errors are diagnostic fault codes emitted by the NVIDIA driver when the GPU or driver stack detects an anomaly. XIDs are the primary mechanism for GPU fault reporting in Linux.

### What XIDs Represent

Each XID maps to a specific fault condition:
- Hardware errors (ECC, thermal, power)
- Driver/firmware state errors (GPU reset, page fault, context switch timeout)
- System-level errors (PCIe link error, GPU fell off the bus)

XIDs are **not application errors**. An XID indicates a fault in the GPU, driver, system integration, or platform — not a bug in CUDA code (though bad code can sometimes trigger driver state errors).

---

### Where XIDs Appear

XIDs are logged to three primary locations:

#### 1. Kernel ring buffer (`dmesg`)
```bash
dmesg | grep NVRM
```

**Example output (illustrative format):**
```
[12345.678] NVRM: Xid (PCI:0000:00:1e.0): 48, pid=2718, name=python, Ch 00000010, intr 00000000
```

**Fields:**
- `Xid`: The error code (e.g., 48)
- `PCI`: The PCIe address of the GPU
- `pid` / `name`: The process that was using the GPU when the error occurred
- `Ch` / `intr`: Channel and interrupt context (low-level driver state)

**When to check:** After any GPU-related crash, hang, or performance anomaly. `dmesg` is the canonical source for XIDs.

---

#### 2. System log (`/var/log/syslog` or `journalctl`)
```bash
grep NVRM /var/log/syslog
# or
journalctl -k | grep NVRM
```

**Use case:** XIDs are also forwarded to syslog by `rsyslogd` or `journald`. This is useful for centralized logging (e.g., forwarding to a SIEM or log aggregator).

---

#### 3. DCGM
```bash
dcgmi dmon -e
```

DCGM monitors XID events in real-time and can forward them to Prometheus (via `dcgm-exporter`) or other telemetry systems. This is the preferred method for **fleet-scale XID monitoring** (see Lab 10).

**Use case:** Continuous monitoring across many nodes. DCGM can alert on XID patterns (e.g., "GPU 3 on node X has reported 5 XIDs in the past hour").

---

### Common XID Codes

Below is a **starter reference** of well-known XID codes. The full table is maintained in `../../reference/xid-table.md` and will be populated with verified examples from Lab 02 and Lab 10.

| XID | Meaning | Typical Cause | Action |
|:----|:--------|:--------------|:-------|
| 13  | Graphics Engine Exception | Invalid GPU operation, corrupted command buffer, driver bug | Check for driver/firmware mismatch, review application logs, consider driver update |
| 31  | GPU Memory Page Fault | Application accessed invalid GPU memory address | Check application for out-of-bounds access, review CUDA error handling |
| 43  | GPU Stopped Responding | GPU hung during execution (timeout) | Check for infinite loop in kernel, thermal throttle, or power issue; may require GPU reset |
| 48  | Double Bit ECC Error (Uncorrectable) | DRAM defect, cosmic ray, severe memory corruption | Check ECC counters, review page retirement status, RMA GPU if errors recur |
| 63  | Row Remapping Event | GPU firmware remapped a failing DRAM row | Normal if infrequent; monitor remap count via `nvidia-smi -q -d ROW_REMAP` |
| 64  | Row Remapping Failure | GPU ran out of spare rows or remapping failed | Check row remap status, likely RMA required |
| 74  | NVLink Error | NVLink CRC error, protocol error, or link training failure | Check NVLink topology, Fabric Manager status, cable seating; re-train links |
| 79  | GPU Fallen Off the Bus | PCIe link lost, GPU no longer enumerable | Check PCIe link status, reseat GPU, inspect for hardware fault (power, cable, riser) |
| 94  | Contained ECC Error | ECC error within a GPU context (application isolated) | Error did not propagate to kernel; check application robustness, monitor ECC rate |
| 95  | Uncontained ECC Error | ECC error propagated to system (may affect kernel or other processes) | Severe; check for memory corruption, RMA GPU if errors continue |

**Note:** This is a **conceptual starter set** based on well-documented XID meanings in NVIDIA driver documentation. The full reference in `../../reference/xid-table.md` will include verified examples as labs encounter them.

---

### Interpreting XIDs: Symptom → Cause → Action

**General approach:**
1. **Identify the XID code** in `dmesg` or DCGM
2. **Look up the code** in `../../reference/xid-table.md`
3. **Check correlated symptoms:**
   - Was the GPU under load? (`nvidia-smi dmon`)
   - Any thermal or power throttle events? (see **Throttle Reasons** below)
   - Any ECC errors? (`nvidia-smi -q -d ECC`)
   - Any NVLink errors? (`nvidia-smi nvlink -e`)
4. **Determine severity:**
   - **Transient (single occurrence):** May be recoverable; monitor for recurrence
   - **Persistent (multiple occurrences over hours/days):** Likely hardware fault; RMA
   - **Fatal (immediate crash/hang):** GPU may be unusable; immediate RMA
5. **Take action** per the table (reset, RMA, tune workload, etc.)

**Example workflow:**
- **Symptom:** Training job crashes after 2 hours
- **XID observed:** XID 48 (double-bit ECC error)
- **Correlated symptom:** `nvidia-smi -q -d ECC` shows uncorrectable error count incremented
- **Cause:** DRAM defect (uncorrectable error)
- **Action:** Reboot node, re-run DCGM Level 3 diag; if errors recur, RMA GPU

---

## Thermal and Power Throttling

GPUs dynamically adjust clock frequencies in response to **thermal** (temperature) and **power** (TDP limit) constraints. Throttling is not always a fault — it is the GPU's self-protection mechanism — but unexpected throttling indicates a problem (inadequate cooling, power delivery, or firmware tuning).

### Querying Throttle Reasons
```bash
nvidia-smi -q -d PERFORMANCE
```

**Output includes:**
- Current clocks (graphics, SM, memory)
- Max clocks (architectural limit)
- Throttle reasons (bitmask of active throttle conditions)

**Example output (illustrative):**
```
Clocks Throttle Reasons
    Idle                        : Active
    Applications Clocks Setting : Not Active
    SW Power Cap                : Not Active
    HW Slowdown                 : Active
    HW Thermal Slowdown         : Not Active
    HW Power Brake Slowdown     : Not Active
    Sync Boost                  : Not Active
```

---

### Throttle Reason Bitmask

Each throttle reason is a flag in a bitmask. Multiple reasons can be active simultaneously.

| Reason | Meaning | Typical Cause | Action |
|:-------|:--------|:--------------|:-------|
| **Idle** | GPU is idle (not running a workload) | Normal | No action required |
| **Applications Clocks Setting** | Clocks locked via `nvidia-smi -ac` or application request | Intentional | Check if clocks were pinned for benchmarking; reset with `nvidia-smi -rac` |
| **SW Power Cap** | Software-imposed power limit (via `nvidia-smi -pl`) | Intentional power capping or cluster policy | Check power limit: `nvidia-smi -q -d POWER`; increase if appropriate |
| **HW Slowdown** | Hardware-imposed throttle due to thermal or power | Temperature above threshold or power draw exceeding TDP | **Critical:** Check temps, cooling, and power delivery |
| **HW Thermal Slowdown** | Temperature-based throttle (GPU too hot) | Insufficient cooling, blocked airflow, failed fans | Check GPU temp (`nvidia-smi -q -d TEMPERATURE`), inspect cooling system |
| **HW Power Brake Slowdown** | Power delivery insufficient or voltage drooping | PSU undersized, 12V rail unstable, PCIe power cable issue | Check power supply, cables, and GPU power connectors |
| **Sync Boost** | Multi-GPU clock synchronization (all GPUs run at slowest GPU's clock) | Mixed GPU clocks in NVLink domain | Ensure all GPUs in NVLink domain have same clock settings |

---

### Symptom → Cause → Action Examples

#### Scenario 1: Thermal Throttle
- **Symptom:** Training throughput drops by 30% after 10 minutes
- **Cause:** `nvidia-smi -q -d PERFORMANCE` shows "HW Thermal Slowdown: Active"; `nvidia-smi -q -d TEMPERATURE` shows GPU temp at 87°C
- **Action:** Check data-center cooling (inlet air temp, HVAC), inspect GPU fans, verify airflow is not blocked, consider reducing ambient temperature or workload intensity

#### Scenario 2: Power Brake
- **Symptom:** GPU crashes during peak load (XID 79 or system hang)
- **Cause:** `nvidia-smi -q -d PERFORMANCE` shows "HW Power Brake Slowdown: Active"; PSU logs show 12V rail voltage drop
- **Action:** Verify PSU is rated for peak GPU load (H100 can draw 700W), check PCIe power cables are fully seated, consider upgrading PSU or distributing load across multiple rails

#### Scenario 3: Sync Boost in Multi-GPU
- **Symptom:** 8-GPU training runs slower than expected; GPUs 0-6 idle while GPU 7 runs hot
- **Cause:** `nvidia-smi -q -d PERFORMANCE` shows "Sync Boost: Active"; GPU 7 is thermally throttled, dragging down other GPUs' clocks
- **Action:** Isolate GPU 7, check cooling, re-balance workload, or replace GPU if thermal issue persists

---

### Monitoring Throttle Events in Real-Time
```bash
watch -n 1 'nvidia-smi -q -d PERFORMANCE | grep -A 10 "Clocks Throttle"'
```

This command polls throttle reasons every second. Use it during burn-in or stress tests to catch transient throttle events.

---

## NVIDIA Bug Report

**`nvidia-bug-report.sh`** is the official NVIDIA script for collecting comprehensive diagnostic data. It is the **required artifact for RMA or support cases**.

### What It Collects

- **System info:** CPU, memory, kernel version, BIOS version
- **GPU inventory:** All enumerated GPUs, PCI topology, firmware versions
- **Driver state:** Loaded kernel modules, driver version, CUDA version, library paths
- **Logs:** `dmesg`, `/var/log/Xorg.0.log`, `/var/log/syslog` (filtered for NVIDIA messages)
- **Configuration:** `xorg.conf`, `modprobe.d/nvidia.conf`, `/proc/driver/nvidia/*`
- **Error counters:** ECC counts, XIDs, page retirement status
- **Runtime state:** Current GPU utilization, running processes, clocks, temps, power

**Output:** A `.gz` tarball (e.g., `nvidia-bug-report.log.gz`) containing all collected data.

---

### When to Run

- **Before opening an NVIDIA support case or RMA:** The support team will request this report
- **After encountering a driver crash or XID event:** Captures the full context of the fault
- **During intermittent issues:** Run immediately after the symptom occurs to capture transient state

---

### How to Run
```bash
sudo nvidia-bug-report.sh
```

**Runtime:** ~30-60 seconds

**Output:** `nvidia-bug-report.log.gz` in the current directory

**Upload:** Attach to NVIDIA support case or include in RMA documentation

**Privacy note:** The report includes system configuration and logs but does not include application code or training data. Review the contents before sharing externally if your system config is sensitive.

---

## GPU Burn: Stress and Correctness Verification

**`gpu-burn`** is a community-developed GPU stress test and correctness checker. Unlike DCGM diagnostics (which focus on hardware faults), `gpu-burn` validates **numerical correctness** under sustained load — catching silent data corruption, thermal instability, and precision errors.

### What It Does

- Runs a large matrix multiplication (GEMM) on the GPU continuously
- Compares GPU results against CPU results (or previously validated GPU results)
- Reports pass/fail based on numerical accuracy (within floating-point tolerance)

**Use case:** Detecting **silent corruption** — errors that do not trigger XIDs or ECC events but produce wrong results (e.g., due to marginal SRAM, voltage droop, or clock instability).

---

### Running GPU Burn

**Container image:** `oguzpastirmaci/gpu-burn`

```bash
# Run for 60 seconds
docker run --rm --gpus all oguzpastirmaci/gpu-burn 60
```

**Output (illustrative):**
```
GPU 0: PASSED
GPU 1: PASSED
GPU 2: PASSED
GPU 3: PASSED
GPU 4: PASSED
GPU 5: PASSED
GPU 6: PASSED
GPU 7: PASSED
```

**Pass:** GPU computed correct results for all iterations
**Fail:** GPU produced incorrect results (mismatch beyond floating-point tolerance)

---

### Interpreting Results

#### Pass
- GPU is stable under sustained compute load
- Numerical precision is within expected bounds
- No silent corruption detected

**Action:** GPU is healthy for production workloads

#### Fail
- GPU produced incorrect results during matrix multiplication
- Possible causes:
  - Marginal SRAM or DRAM (errors not detected by ECC)
  - Clock instability (overclocking, voltage droop)
  - Thermal instability (GPU throttled mid-computation)
  - Firmware bug

**Action:**
1. Re-run `gpu-burn` to confirm failure is reproducible
2. Check for thermal/power throttle (`nvidia-smi -q -d PERFORMANCE`)
3. Run DCGM Level 3 diagnostics to check for hardware faults
4. If failure persists and no other symptoms, consider firmware update or RMA

---

### Symptom → Cause → Action Example

- **Symptom:** Training loss diverges after 1 hour; loss values become NaN
- **Cause:** `gpu-burn` reports FAIL on GPU 3; no XID or ECC errors
- **Action:** Isolate GPU 3, re-run DCGM Level 3 diag, check for marginal hardware; RMA if `gpu-burn` consistently fails

---

### GPU Burn vs. DCGM Diagnostics

| Tool | Focus | Runtime | What It Catches |
|:-----|:------|:--------|:----------------|
| **DCGM Level 1-4** | Hardware faults | 10 sec - 4 hours | ECC errors, XIDs, PCIe/NVLink failures, thermal/power issues |
| **gpu-burn** | Numerical correctness | 1-60 minutes | Silent corruption, marginal hardware, precision errors |

**Best practice:** Run both. DCGM catches hardware faults; `gpu-burn` catches silent corruption. Use DCGM for pre-deployment burn-in and `gpu-burn` for periodic correctness validation.

---

## Summary: Diagnostic Decision Tree

Use this decision tree to choose the right diagnostic tool:

1. **Quick smoke-test (before a job):**
   - → `dcgmi diag -r 1` (30 sec)

2. **Performance regression or post-reboot validation:**
   - → `dcgmi diag -r 2` (5 min)

3. **Intermittent crashes or pre-production burn-in:**
   - → `dcgmi diag -r 3` (30 min)

4. **Suspected silent corruption or NaN loss:**
   - → `gpu-burn` (1-60 min)

5. **Preparing for RMA or deep investigation:**
   - → `dcgmi diag -r 4` (1-4 hours) + `nvidia-bug-report.sh`

6. **Investigating a specific XID or ECC event:**
   - → Check `dmesg | grep NVRM`
   - → Query ECC/page retirement: `nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAP`
   - → Look up XID in `../../reference/xid-table.md`

7. **Investigating performance throttle:**
   - → `nvidia-smi -q -d PERFORMANCE` (check throttle reasons)
   - → `nvidia-smi -q -d TEMPERATURE,POWER` (check temps and power draw)

---

## Cross-References

- **XID Error Reference Table:** `../../reference/xid-table.md` (populated in Task 2.2, verified in Lab 02 and Lab 10)
- **Driver and CUDA stack:** `docs/part1-single-node/02-driver-cuda-stack.md`
- **Fleet-scale observability:** `docs/part3-clustering-execution/10-fleet-observability.md`
- **DCGM metrics pipeline:** `docs/toolkit/T1-monitoring-inventory.md`

---

## Practice

See **Lab 02** (driver and CUDA health diagnostics) and **Lab 10** (fleet-scale health monitoring with DCGM and Prometheus) for hands-on exercises with these tools.
