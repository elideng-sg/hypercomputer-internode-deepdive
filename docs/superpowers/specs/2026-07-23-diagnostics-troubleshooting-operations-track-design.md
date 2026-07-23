# Design Spec: Diagnostics, Troubleshooting & Operations track (Part V)

**Date:** 2026-07-23 (rev 1)
**Author:** elideng-sg (with Claude Code)
**Status:** Draft — pending user review
**Sibling spec:** [`2026-07-23-three-node-scaling-enrichment-design.md`](2026-07-23-three-node-scaling-enrichment-design.md) (independent; ships separately)
**Parent design:** [`2026-07-21-hypercomputer-internode-deepdive-design.md`](2026-07-21-hypercomputer-internode-deepdive-design.md)

---

## 1. Purpose & motivation

The guide is used to **ramp engineers on design, troubleshooting, and implementation** for **both GCP A3 infrastructure and the NVIDIA toolchain**. A review of the existing 11 labs found the guide is strong at *inventory + mechanism + one honest measurement*, but has a systematic gap in the exact skill this track targets:

> **Almost every lab walks the *healthy path* and observes.** Only lab-10 runs the loop that actually ramps a troubleshooter: **symptom → hypothesis → which tool → read the output → root cause → fix.** You cannot build diagnostic skill by only ever looking at working systems.

Secondary findings this track also closes:
- **Marquee NVIDIA tools are named in prose but never run:** `ncu` (only ever fails on permissions), HTA, `nccl-tests` binaries, `dcgmi dmon`/health, `perftest`/`ethtool`, `nvidia-bug-report.sh`, Grafana/PromQL.
- **The reference "knowledge" backbone is placeholder:** `reference/nccl-tunables.md` ("To be filled…"), `reference/tool-cheatsheets.md` (4× "Commands to be filled"). For a guide whose purpose includes NVIDIA *knowledge*, these are the cheapest high-leverage fixes.
- **Two data-integrity bugs** to fix in passing: `assets/lab-01/devquery.txt` reproduces a garbage "global memory: 3242481418240 MBytes" reading uncritically; `labs/lab-05/README.md:24` says the image has no `ip` while `run.sh:47` calls `ip -br link`.

This track adds a **scenario-based diagnostics, troubleshooting, and operations layer** — a new **Part V** plus a filled reference backbone — covering the five areas the user prioritized:

1. **Inter-node comms debugging** (highest value)
2. **Cluster / job failures**
3. **Single-GPU health**
4. **NVIDIA tool depth**
5. **Performance monitoring & routine (day-2) diagnostics** — steady-state ops, not just fault injection

**Guiding rule (inherited & sharpened):** every lab is a *scenario* — the engineer is handed a symptom or an operational task and must reach a root cause / decision using the right tool. No healthy-path-only exercises. Every tool is **used to reach a conclusion**, never merely demonstrated.

---

## 2. Design principles

- **The triage loop is the unit.** Generalize lab-10's proven *symptom → signature → root-cause → fix* tree into a methodology doc (doc-16) that every scenario lab instantiates.
- **Break it safely, on the live cluster.** Faults are injected at the **job/pod/env level** (bad env var, memory blowup, mismatched collective, artificial delay, killed rank). **No node is drained or deleted** — Flex capacity stays held (the "always hold the GPU" posture). Every scenario is reversible; holders untouched.
- **Constraints become lessons.** Managed GKE blocks `dmesg`; that is *itself* the lesson — teach where XIDs actually surface on GCP (Node Problem Detector → Cloud Logging, `DCGM_FI_DEV_XID_ERRORS`) vs. bare-metal `dmesg`/`nvidia-bug-report.sh`. Same for `ncu` privilege (`securityContext`/`CAP_SYS_ADMIN` + `--target-processes`).
- **Tools in anger, not on parade.** `ncu` → single-GPU + perf labs; HTA → perf lab; `nccl-tests` binaries + `ethtool` → comms lab; `dcgmi dmon`/health + PromQL/Grafana → ops lab.
- **Two lenses on every failure:** the **GCP/GKE** view (kubectl, events, Cloud Logging, Kueue/JobSet status, DWS) *and* the **NVIDIA** view (nvidia-smi/DCGM/NCCL/Nsight), taught side by side so engineers learn which lens answers which question.
- **Real vs. read-this-broken-output.** Prefer live-injected faults; where a fault is unsafe or unreproducible on this cluster (e.g. a real double-bit ECC / XID 79 fall-off-bus), provide a **curated captured/synthetic signature** clearly labeled as such, with the decode + action — never fabricated as if live.
- **All provenance logged** to `VERIFICATION.md`, cluster/context named, per the existing discipline.

---

## 3. What is added

### Part V — Operations, Diagnostics & Troubleshooting (new doc set)

| Doc | Title | Role |
| :--- | :--- | :--- |
| **doc-16** | The diagnostic method (triage framework) | Generalizes lab-10's loop: how to localize *any* GPU/cluster problem; the decision tree; which lens (GCP vs NVIDIA) answers which question; how to read a signature. Every lab-1x links here. |
| **doc-17** | Single-GPU & node health diagnostics | XID on managed GKE, throttling, ECC/RAS, DCGM diag levels, `nvidia-bug-report`. |
| **doc-18** | Inter-node comms troubleshooting | Localizing a slow/broken collective; NCCL hang/timeout; env/MTU/rail faults; the debug decision tree. |
| **doc-19** | Cluster & job failure triage | Pending gangs, OOM, crashloop, Kueue inadmissible, restart/preemption. |
| **doc-20** | Performance monitoring & day-2 operations | Baselines, regression detection, DCGM profiling fields, Grafana/PromQL alerting, routine health runbook. |

### Scenario labs (each = one triage loop; tool-depth folded in)

| Lab | Scenario(s) — engineer is handed a symptom/task | NVIDIA + GCP tools finally exercised |
| :--- | :--- | :--- |
| **lab-14** Single-GPU health triage | "This GPU is throttling / suspected bad" — reproduce throttle under `gpu-burn` load and capture clocks + **throttle reasons**; query **ECC / row-remap / retired pages**; run **DCGM diag** and read a **Warning/Fail**; decode an **XID** via the GKE path; produce `nvidia-bug-report.sh` | `nvidia-smi -q -d` under load, `nvidia-smi dmon`, `dcgmi diag -r 3`, `dcgmi dmon -e`, Cloud Logging XID, `nvidia-bug-report.sh`; **`ncu`** privileged kernel analysis (the half of profiling lab-03 never did) |
| **lab-15** Inter-node comms debugging | "The all-reduce is slow / hangs" — **localize** it (transport? one slow rail? wrong `NCCL_SOCKET_IFNAME`? MTU black-hole? straggler rank?); reproduce an **init-hang** and a **collective timeout/watchdog**; broken-vs-healthy transport | `NCCL_DEBUG=INFO/WARN` (INIT,NET,**GRAPH**), `TORCH_NCCL_*` timeouts, **`nccl-tests` `all_reduce_perf`** binary, **`ethtool`** counters/drops, per-rank timing |
| **lab-16** Cluster & job failure triage | "My job won't run / crashed / is stuck" — **OOMKilled** (exit 137), **crashloop**, **gang stuck-Pending** root-cause, **Kueue inadmissible** (flavor mismatch / cohort borrowing), **checkpoint→restart** via `failurePolicy.maxRestarts`, **pod eviction/reschedule** | `kubectl` events/logs/`describe`, JobSet `failurePolicy`, Kueue `Workload` conditions, DWS status, Cloud Logging |
| **lab-17** Performance monitoring & day-2 ops | *Not a fault* — steady-state: capture a **performance baseline**, stand up a **Grafana dashboard + PromQL alerts**, then **detect a regression** (silent throttle / a straggler node / NIC degradation) from monitoring *before* it becomes a crash; a **routine health-check runbook** | **dcgm-exporter** `DCGM_FI_PROF_*` (SM/tensor occupancy, PCIe/NVLink BW), **PromQL/MQL**, **Grafana** dashboards + alert rules, **HTA** comm/compute-overlap on a real trace |

### Reference backbone — fill the stubs (verified placeholders today)

- `reference/xid-table.md` — real XID catalog: code → meaning → likely cause → action, cross-linked from lab-14/doc-17.
- `reference/nccl-tunables.md` — the tunables actually varied in labs (`NCCL_ALGO/PROTO`, `NCCL_SOCKET_IFNAME`, `NCCL_NSOCKS_PERTHREAD`, `SOCKET_NTHREADS`, `NCCL_DEBUG_SUBSYS`, timeouts) with what each did to the measured numbers.
- `reference/tool-cheatsheets.md` — populate every section from the commands the labs actually run (nvidia-smi, dcgmi, nsys/ncu, nccl-tests, ethtool, perftest, kubectl/Kueue/JobSet).
- `reference/driver-matrix.md` — extend as versions are captured across both clusters.

### Quick-win fixes (independent, ship first)

- Correct the `assets/lab-01/devquery.txt` garbage-memory reading (annotate as a *caught bad reading* — turn the bug into a mini inspection lesson rather than hiding it).
- Resolve the lab-05 `ip`-availability inconsistency (README vs `run.sh:47`).

---

## 4. Repository changes

```
docs/
  part5-operations-diagnostics/
    16-diagnostic-method.md            (NEW — triage framework, hub doc)
    17-single-gpu-node-health.md       (NEW)
    18-internode-comms-troubleshooting.md (NEW)
    19-cluster-job-failure-triage.md   (NEW)
    20-performance-monitoring-day2-ops.md (NEW)
labs/
  lab-14-single-gpu-health-triage/     (NEW)
  lab-15-internode-comms-debug/        (NEW)
  lab-16-cluster-job-failure-triage/   (NEW)
  lab-17-perf-monitoring-day2-ops/     (NEW)
manifests/
  fault-oom.yaml, fault-crashloop.yaml, fault-nccl-hang.yaml (extend),
  fault-straggler.yaml, kueue-inadmissible.yaml,
  grafana-dashboard.json, prometheus-alert-rules.yaml        (NEW as needed)
scripts/
  lib_capture.sh (already parametrized by sibling spec — reuse),
  inject_fault.sh / triage helpers                            (NEW)
reference/
  xid-table.md, nccl-tunables.md, tool-cheatsheets.md, driver-matrix.md (FILL)
assets/lab-14..17/                     (NEW captured signatures, dashboards, traces)
docs/00-guide-overview.md, README.md   (edit: add Part V + labs 14-17)
VERIFICATION.md                        (append)
```

---

## 5. Capture / execution plan

Per scenario: (1) capture the **healthy baseline**, (2) **inject** the fault at job/pod/env level (or load a labeled curated signature where unsafe), (3) walk the **triage loop** capturing each tool's output, (4) apply the **fix** and re-capture green, (5) **revert** fully. Cluster chosen per scenario (single-GPU health → either; cluster-failure & straggler → prefer 3-node `asia-east1-c` for room), always labeled. All output through `lib_capture.sh` → `assets/` + `VERIFICATION.md`.

Build order (each independently shippable):
1. **Quick wins** (data-bug fixes) + **reference backbone fill** — immediate, no cluster needed for the fills beyond what's captured.
2. **doc-16 methodology** + **lab-15 inter-node comms** (highest-value scenario).
3. **lab-14 single-GPU health** + **lab-16 cluster/job failures**.
4. **lab-17 perf monitoring & day-2 ops** (Grafana/PromQL/HTA).
5. Part V docs finalized + README/overview wiring.

---

## 6. Correctness & verification principles (inherited)

- Never claim a signature/number not read off a live run — or, if curated/synthetic (unsafe-to-inject faults), **label it as such** with decode + action.
- Flex-safe: no node drain/delete; all faults reversible; holders untouched.
- Both-lens honesty: show the GCP *and* NVIDIA view of each failure.
- Full provenance in `VERIFICATION.md`, cluster/context named.

---

## 7. Explicitly out of scope (YAGNI)

- Destructive/hardware faults that require node deletion or risk Flex capacity (real fall-off-bus XID, physical NVLink pull) — covered as **curated labeled signatures**, not live.
- Enabling TCPX/RDMA (owned by neither this nor the scaling spec).
- Chaos-engineering automation / synthetic load frameworks beyond simple injectors.
- Rewriting existing labs 01-11 (only the two data-bug fixes + cross-links into Part V).
- Alerting *delivery* integrations (PagerDuty/Slack) — alert *rules* only.

---

## 8. Success criteria

1. Every lab-1x is a **scenario** ending in a root cause or an operational decision — zero healthy-path-only exercises.
2. Each of the five priority areas has at least one exercised triage loop, with the **NVIDIA tool used to reach the conclusion** (`ncu`, HTA, `dcgmi dmon`, `nccl-tests`, `ethtool`, PromQL/Grafana all finally run in anger).
3. The **XID / throttle / OOM / hang / straggler / inadmissible** signatures that today's docs only *describe* are actually **produced and read** (or curated + labeled where unsafe).
4. `reference/xid-table.md`, `nccl-tunables.md`, `tool-cheatsheets.md` are **filled** from real lab usage — no placeholders remain.
5. doc-20 gives a runnable **day-2 monitoring** setup (dashboard + alert rules + baseline + regression detection), not just fault triage.
6. All Flex nodes stay held; every scenario reversible; provenance complete.
7. New Part V matches the repo's rigor, tone, mermaid style, and cross-linking; the two data-integrity bugs are fixed.
