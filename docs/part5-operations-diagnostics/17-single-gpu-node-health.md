# 17 — Single-GPU & Node Health: Reading a GPU Under Load, and *Why* It's Slow

## Overview

Most of this guide walks the healthy path: a GPU exists, it benchmarks, move on. Day-2 operations asks a harder question — *the GPU is up, but the job is slow; is the hardware healthy under load, and if it's throttling, why?* This document is the **GPU-die / node-layer** instantiation of [doc-16](16-diagnostic-method.md)'s triage method. Its central discipline: **read health against a baseline, and read severity from the clock delta — never from a boolean.**

The trap this layer sets is a false alarm. `nvidia-smi` will happily report `SW Power Cap: Active` on a perfectly healthy GPU running flat-out — because that *is* the power governor doing its job at TDP. The skill is knowing that the throttle **reason** is a starting point, not a verdict; the **magnitude of the clock drop** is the verdict. Everything below is about establishing the baseline, reading the under-load delta, and inducing a *controlled* throttle so you know what a real one looks like.

**What you'll learn:**
- The **idle → under-load delta** — what `nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE` and `nvidia-smi dmon` should show, and how the baseline turns a raw number into a signal
- The **throttle-reason taxonomy** — `SW Power Cap` vs. `HW Slowdown`/`HW Thermal`/`HW Power Brake` vs. `SW Thermal` vs. `Sync Boost` — which are benign-at-TDP and which are always a fault
- **Reading severity from the clock, not the flag** — and a *deterministic, reversible* power-cap experiment that produces a real throttle on demand
- The **profiling-permission wall** (`ERR_NVGPUCTRPERM`) and how a privileged context reads what an ordinary pod can't
- **Where an XID actually surfaces on managed GKE** — NPD → Cloud Logging, *not* the managed `dcgm-exporter` — and the DCGM **version-matching** trap on newer drivers

**Prerequisites:** [doc-16](16-diagnostic-method.md) (the triage loop, crash-vs-hang timing, the two lenses); [doc-02](../part1-single-node/02-drivers-cuda-install-troubleshooting.md) (health/`dcgmi`/XID basics); [doc-03](../part1-single-node/03-single-gpu-execution-and-profiling.md) (single-GPU profiling and the `ncu` permission wall).

**Instantiated by:** [lab-14](../../labs/lab-14-single-gpu-health-triage/) — a live idle baseline, an under-load delta, a deterministic 700→200 W throttle, the privileged `ncu` rerun that completes lab-03, and the XID surface, all on the asia-east1-c 3-node cluster (driver 580.126.20).

---

## Step 1 — Establish the idle baseline (the reference read)

A health number means nothing on its own; it means something *relative to idle*. Capture the baseline first:

```bash
nvidia-smi -i 0 -q -d ECC,PERFORMANCE,CLOCK,POWER,TEMPERATURE,ROW_REMAPPER
```

> **First trap (gotcha G10):** the group tokens are exact. `ROW_REMAP` and `PCIE` are **not** valid `-d` groups — one bad token rejects the *entire* comma list. Use `ROW_REMAPPER`; there is no `PCIE` group.

What a healthy idle H100 looks like (lab-14, device 0):

```
Idle throttle reason : Active          SW Power Cap : Not Active     HW Slowdown : Not Active
SM clock             : 345 MHz         Power draw   : 72.12 W        Temp        : 36 C
ECC (SRAM+DRAM, corr+uncorr) : all 0   Remapped Rows / Failure      : 0 / No
```

Two things to bank here. The persistent-error counters — **ECC** (correctable/uncorrectable, SRAM and DRAM) and **remapped rows** — are the ones you read *cold*, at idle: a nonzero uncorrectable count or a `Remapping Failure Occurred: Yes` is a hardware verdict independent of load. Everything else (clocks, power, throttle reasons) only becomes meaningful under load, which is Step 2.

---

## Step 2 — Read the under-load delta

Put the GPU under sustained compute (lab-14 drives fp16 GEMMs) and re-read the same groups plus a time series:

```bash
nvidia-smi dmon -s pucvmet -c 20        # power/util/clocks/violations/mem/temp over time
nvidia-smi -i 0 -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE
```

The idle→load delta from lab-14 (same device 0, 700 W limit):

| Signal | Idle | Under load |
|---|---|---|
| `Idle` reason | `Active` | `Not Active` |
| `SW Power Cap` | `Not Active` | **`Active`** |
| `HW Slowdown` / `HW Thermal` | `Not Active` | `Not Active` |
| SM clock | 345 MHz | **1365 MHz** |
| Power (avg / inst) | 72 W / 72 W | **697 W / 701 W** |
| Temp | 36 °C | 63 °C |

**This is the false alarm.** `SW Power Cap: Active` under full load is **not a fault** — it is the governor holding the card at its 700 W TDP, which is *precisely why* the SM clock sits at 1365 MHz instead of the 1980 MHz max application clock. A healthy flat-out GPU trades some boost headroom for its power budget. `HW Slowdown`/`HW Thermal`/`HW Power Brake` all reading `Not Active`, temp well under the ~85 °C slowdown threshold, and no ECC/replay growth is the "healthy under load" signature. The `dmon` `pviol` column (power-violation %) ticking up is the same story in the time domain: time spent at the power cap, not a fault.

---

## Step 3 — Read severity from the clock, not the flag

If a throttle reason *is* active, the question is **how much is it costing you** — and that is a clock measurement, not a boolean. lab-14 makes this concrete with a **deterministic, reversible** experiment: cap device 0's power limit to its 200 W floor, re-load it, and read the result:

```
Current Power Limit : 200.00 W   (Default 700.00 W, Min 200.00 W)
SW Power Cap        : Active
SM clock            : 345 MHz     ← collapsed from 1365 MHz under the same load
```

Same throttle *reason* as the healthy Step-2 case (`SW Power Cap: Active`), utterly different *severity*: the SM clock has collapsed to the idle floor. **That** is a throttle worth chasing. Then it is restored — `nvidia-smi -i 0 -pl 700` — and the card reads `700.00 W` again. (On real hardware the equivalent severe signatures are `HW Thermal Slowdown` on a cooling failure or `HW Power Brake` on a PSU/rail problem; you read them the same way — by the clock they impose.)

> **Throttle-reason cheat sheet.** `SW Power Cap` → governor at TDP, benign unless the clock drop is large (then: check the power limit, `nvidia-smi -q -d POWER`). `HW Thermal Slowdown` → cooling/thermal, always investigate (check `TEMPERATURE`, airflow). `HW Power Brake` → external power event, always investigate. `SW Thermal Slowdown` → driver-side thermal management. `Sync Boost` → clocks pinned to a slower peer GPU in a sync-boost group. **Every one is read by its clock delta against the idle/boost baseline from Steps 1–2.**

---

## Step 4 — Profiling under load needs a privileged context (`ERR_NVGPUCTRPERM`)

When "why is it slow?" needs *per-kernel* truth rather than device-level telemetry, you reach for `ncu` — and on managed GKE hit the wall doc-03/lab-03 documented:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

GKE's COS loads the driver with `NVreg_RestrictProfilingToAdminUsers=1`, so hardware perf counters require **`CAP_SYS_ADMIN`**. In a **privileged** pod, `ncu` collects real metrics (lab-14, three cuBLAS `nvjet` kernels):

```
SM Frequency 1.36 GHz   Compute (SM) Throughput 73.44 %   DRAM Throughput 16.34 %   Occupancy 14.14 %
```

Compute-bound, as a large GEMM should be. Two operational notes: `ncu --set full` profiles *every* kernel with the full metric set — bound it with `--launch-count N --launch-skip M` or it looks like a hang (gotcha G5); and `nsys` does **not** need this capability (it uses CUPTI activity tracing, not perf counters), so reach for `nsys` first when a privileged context isn't available.

---

## Step 5 — Where an XID actually surfaces (and the DCGM version trap)

XIDs are the driver's hardware/driver fault codes ([xid-table.md](../../reference/xid-table.md)). The instinct is `dmesg | grep Xid` — but on managed GKE:

- **Unprivileged pods can't read `dmesg`** at all (gotcha G6) — the kernel ring buffer needs `CAP_SYSLOG`/host access. A *privileged* pod can (lab-14 reads driver `580.126.20`, no Xid), but **never build production tooling on that**.
- **The managed `dcgm-exporter` does *not* export `DCGM_FI_DEV_XID_ERRORS`** (gotcha G11). Its default field set is FB/temp/util/power/clock + the PROF_* pipe/DRAM/NVLink/PCIe metrics — **no XID field**. The common "just scrape `DCGM_FI_DEV_XID_ERRORS`" advice does not hold out of the box; you'd have to run your own exporter with a custom field list.
- **The real GKE-native path is Node Problem Detector → Cloud Logging** — query the node's logs for `NVRM: Xid`, and NPD also raises a node **condition/event** (`kubectl get events`). That is the query you actually run in an incident.

And the diagnostic that *should* close the loop, `dcgmi diag -r N`, hits a **version-matching wall** on newer drivers. On this cluster's R580/CUDA-13 driver, the DCGM 3.3.8 image fails `Detected unsupported Cuda version` (G12) and the 4.4.0 image can't load `plugins/cuda13/` (G13) — **no stock public DCGM diag image runs the deep suite on R580 today.** The lesson generalizes past this one driver: **pin your DCGM image — daemon *and* diag plugins — to the node's CUDA generation, and verify `dcgmi diag` actually runs there before you put it in a runbook.**

---

## Signature catalog — the single-GPU/node layer

Extends the [doc-16 catalog](16-diagnostic-method.md#a-generalized-signature-catalog). "First check" is the fastest lens for that row.

| Symptom | Signature (where you read it) | Likely root cause | First check |
|---|---|---|---|
| Job slow, GPU pegged at 100 % | `SW Power Cap: Active`, SM clock **well below** boost | Power-limited (governor at TDP, or a *lowered* limit) | `nvidia-smi -q -d POWER` — is `Current Power Limit` < default? |
| Job slow, GPU hot | `HW Thermal Slowdown: Active`, temp near ~85 °C | Cooling/airflow failure | `nvidia-smi -q -d TEMPERATURE`; node/rack thermals |
| Intermittent wrong results / crashes | ECC **uncorrectable** > 0, or remapped-row growth | Failing HBM | idle `-q -d ECC,ROW_REMAPPER`; drain node |
| Clocks pinned low across GPUs | `Sync Boost: Active` | One slow peer dragging a sync-boost group | per-GPU `dmon` — find the laggard |
| App reports a GPU fault / fell off the bus | `NVRM: Xid <n>` in **Cloud Logging** (NPD); node event | Xid-specific (see [xid-table.md](../../reference/xid-table.md)) | Cloud Logging `NVRM: Xid`; `kubectl get events` — **not** the managed exporter |
| `ncu` refuses to profile | `ERR_NVGPUCTRPERM` | No `CAP_SYS_ADMIN` for perf counters | privileged pod + `--target-processes all`, or use `nsys` |
| `dcgmi diag` won't run | `unsupported Cuda version` / missing `plugins/cuda13/` | DCGM image mismatched to driver CUDA | pin DCGM image to the node's CUDA generation |

---

## Key takeaways

- **Baseline first.** A health number is a signal only against idle; capture ECC and remapped-rows *cold*, clocks/power/temp *under load*.
- **`SW Power Cap: Active` is usually the governor, not a fault.** Read severity from the **clock delta**, never from the throttle boolean.
- **A controlled fault teaches the signature.** A reversible power cap (700→200 W → SM 1365→345 MHz) shows exactly what a real severe throttle looks like — inject faults you can guarantee to revert.
- **Profiling under load needs `CAP_SYS_ADMIN`** on managed GKE (`ERR_NVGPUCTRPERM`); `nsys` doesn't.
- **XIDs surface via NPD → Cloud Logging on managed GKE**, not the managed `dcgm-exporter`; and DCGM diag must match the node's CUDA generation.

**Next:** [lab-14](../../labs/lab-14-single-gpu-health-triage/) runs all of this live. Then [doc-18](18-internode-comms-troubleshooting.md) moves up a layer to the NIC/fabric.
