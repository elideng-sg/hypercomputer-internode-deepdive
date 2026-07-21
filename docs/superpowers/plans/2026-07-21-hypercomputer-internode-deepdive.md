# hypercomputer-internode-deepdive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, hands-on guide to the NVIDIA GPU + AI-infrastructure stack (single node → inter-node → cluster → platform reference architectures), where every mechanism doc is paired with a lab executed live on the real 2-node A3 H100 cluster and documented with actual captured output.

**Architecture:** A documentation-and-labs repository. Reusable capture/plot scripts (`scripts/`) run workloads on the live GKE cluster and emit artifacts into `assets/`; mechanism docs (`docs/`) and lab READMEs (`labs/`) embed those real artifacts; `VERIFICATION.md` records provenance for every run. Content is organized in 4 parts plus a cross-cutting toolkit reference. Each phase below is independently shippable.

**Tech Stack:** GKE 1.33, `kubectl`, JobSet, Kueue, Kubeflow `mpi-operator` (for `nccl-tests`), NVIDIA container images (CUDA, PyTorch NGC), NCCL + nccl-tests, DCGM / dcgm-exporter, Nsight Systems/Compute, nvbandwidth, gpu-burn, Prometheus + Grafana (kube-prometheus-stack), Python 3 (matplotlib, pandas) for parsing/plotting, Bash.

## Global Constraints

Copied verbatim from the spec (`docs/superpowers/specs/2026-07-21-hypercomputer-internode-deepdive-design.md`). Every task's requirements implicitly include these.

- **Execute before documenting.** No number, command output, or plot is written into a doc unless it was produced by a real run on the live cluster and saved under `assets/`. Never fabricate measured values.
- **Verify hardware claims against running nodes.** Machine-family networking (TCPX/TCPXO/RDMA), NIC type/count, NVLink gen, SM/clock/Tensor-Core specs, and the actual NCCL data path are confirmed on the nodes, not assumed.
- **Preserve GPU holds.** Never delete or scale down the DWS capacity holders; labs run alongside them. Cluster mutations are additive and reversible; each lab documents its teardown.
- **Provenance.** Every runnable lab appends an entry to `VERIFICATION.md`: what ran, UTC timestamp (from the run output), node names, and the artifact path.
- **Portability & product attribution.** GCP-specific steps are flagged with the generic/cross-product equivalent, and each concept is attributed to the correct GCP machine family (A3 High/Mega/Ultra, A4, A4X) per the §1 product-mapping table.
- **Honesty for Part IV.** DGX/HGX system software, BlueField, and Spectrum-X are NOT on this cluster. Part IV is knowledge-first; it must never imply that hardware was run here, and must separate measured facts from reference knowledge.
- **Cluster identity (do not hardcode elsewhere):** project `hdlab-elideng`, cluster `hypercomputer-a3-cluster`, region `us-central1`, pool `a3-h100-dws-pool`, nodes `a3-highgpu-8g` (8×H100 80GB each; node = HGX H100 baseboard).

---

## File Structure

Files created by this plan (responsibilities in the spec §3). Grouped by phase.

**Scaffolding / shared:**
- `docs/00-guide-overview.md` — journey, cluster-at-a-glance, portability & product mapping, how to run labs safely.
- `scripts/lib_capture.sh` — shared Bash helpers: run-and-tee to `assets/`, append to `VERIFICATION.md`, kubectl-exec-into-GPU-pod, node enumeration.
- `scripts/lib_plot.py` — shared Python: CSV→matplotlib line/bar plots with consistent styling; save PNG under `assets/`.
- `scripts/gpu_pod.yaml` — reusable single-GPU debug Pod (requests `nvidia.com/gpu: 1`, `/dev/shm`, tolerations) used by several labs.
- `VERIFICATION.md` — provenance log (append-only).
- `reference/glossary.md`, `reference/xid-table.md`, `reference/nccl-tunables.md`, `reference/driver-matrix.md`, `reference/tool-cheatsheets.md`, `reference/reference-arch-cheatsheet.md`.

**Toolkit reference (Phase 1):** `docs/toolkit/T1..T6-*.md`

**Part I (Phase 2):** `docs/part1-single-node/01..04-*.md`; `labs/lab-01..04-*/`

**Part II (Phase 3):** `docs/part2-inter-node/05..06-*.md`; `labs/lab-05..06-*/`

**Part III (Phase 4):** `docs/part3-clustering-execution/07..10-*.md`; `labs/lab-07..10-*/`; `manifests/` (JobSet, MPIJob, PyTorch JobSet, dcgm-exporter, kube-prometheus values).

**Part IV (Phase 5):** `docs/part4-platform-reference-arch/11..14-*.md`; `labs/lab-11-platform-compare/`

---

## Conventions used by every LAB task

Each `labs/lab-XX-*/` contains:
- `run.sh` — sources `scripts/lib_capture.sh`, runs the workload, tees raw output to `assets/lab-XX/`, appends provenance.
- manifest(s) (`.yaml`) or references into `manifests/`.
- `parse.py` (when the lab produces tabular data) — reads raw output, writes `assets/lab-XX/*.csv`, calls `lib_plot.py`.
- `README.md` — objective, prerequisites, **runnable-here vs read-only** banner (Part IV only), numbered steps, embedded real output/plots, interpretation, teardown, links to its `docs/` layer + `docs/toolkit/` docs.

**Standard LAB task step sequence** (the "test cycle" for a lab):
1. Write the manifest/script (exact content in the task).
2. Validate locally: `kubectl apply --dry-run=server -f <manifest>` (expected: `configured`/`created (dry run)`), or `bash -n run.sh`.
3. Execute on the live cluster via `run.sh`; confirm the workload reached `Completed`/expected exit.
4. Verify artifacts: the expected files exist under `assets/lab-XX/` and are non-empty and sane (assertion command given per task).
5. Write the mechanism doc + lab `README.md`, embedding the **actual** captured values/plots (reference the artifact paths).
6. Append the provenance line to `VERIFICATION.md`.
7. Commit.

---

## Phase 0 — Repository scaffolding

### Task 0.1: Shared capture library

**Files:**
- Create: `scripts/lib_capture.sh`
- Create: `VERIFICATION.md`

**Interfaces:**
- Produces: Bash functions `cap_run <label> <outfile> -- <cmd...>` (runs cmd, tees stdout+stderr to `assets/<label>/<outfile>`, returns cmd exit code), `cap_verify_provenance <lab> <artifact> <nodes> <note>` (appends a row to `VERIFICATION.md`), `cap_nodes` (echoes the 2 GPU node names), `cap_gpu_exec <pod> -- <cmd...>` (kubectl exec helper).

- [ ] **Step 1: Write `scripts/lib_capture.sh`**

```bash
#!/usr/bin/env bash
# Shared capture helpers. Source this from every lab's run.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$REPO_ROOT/assets"
VERIF="$REPO_ROOT/VERIFICATION.md"

cap_nodes() {
  kubectl get nodes -l cloud.google.com/gke-nodepool=a3-h100-dws-pool \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
}

cap_run() { # cap_run <label> <outfile> -- <cmd...>
  local label="$1" outfile="$2"; shift 2; [ "$1" = "--" ] && shift
  mkdir -p "$ASSETS/$label"
  local path="$ASSETS/$label/$outfile"
  echo "# cmd: $*" | tee "$path"
  "$@" 2>&1 | tee -a "$path"
}

cap_gpu_exec() { # cap_gpu_exec <pod> -- <cmd...>
  local pod="$1"; shift; [ "$1" = "--" ] && shift
  kubectl exec "$pod" -- bash -lc "$*"
}

cap_verify_provenance() { # <lab> <artifact> <nodes> <note>
  local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '| %s | %s | %s | %s | %s |\n' "$ts" "$1" "$2" "$3" "$4" >> "$VERIF"
}
```

- [ ] **Step 2: Write `VERIFICATION.md` header**

```markdown
# Verification Log

Provenance for every live run. Appended automatically by `scripts/lib_capture.sh`.

| UTC timestamp | lab | artifact | nodes | note |
| :--- | :--- | :--- | :--- | :--- |
```

- [ ] **Step 3: Syntax-check**

Run: `bash -n scripts/lib_capture.sh && echo OK`
Expected: `OK`

- [ ] **Step 4: Smoke-test node enumeration on the live cluster**

Run: `source scripts/lib_capture.sh && cap_nodes`
Expected: two node names ending `-dws-pool-*`.

- [ ] **Step 5: Commit**

```bash
git add scripts/lib_capture.sh VERIFICATION.md
git commit -m "scaffold: shared capture library and verification log"
```

### Task 0.2: Shared plotting helper + reusable GPU pod

**Files:**
- Create: `scripts/lib_plot.py`
- Create: `scripts/gpu_pod.yaml`

**Interfaces:**
- Produces: `lib_plot.py` CLI `python3 scripts/lib_plot.py --csv <f> --x <col> --y <col[,col]> --out <png> [--logx] [--title T] [--xlabel L] [--ylabel L] [--kind line|bar]`.

- [ ] **Step 1: Write `scripts/lib_plot.py`**

```python
#!/usr/bin/env python3
import argparse, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--logx", action="store_true"); ap.add_argument("--kind", default="line")
    ap.add_argument("--title", default=""); ap.add_argument("--xlabel", default="")
    ap.add_argument("--ylabel", default="")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    fig, ax = plt.subplots(figsize=(8,5))
    for col in a.y.split(","):
        if a.kind == "bar": ax.bar(df[a.x].astype(str), df[col], label=col)
        else: ax.plot(df[a.x], df[col], marker="o", label=col)
    if a.logx: ax.set_xscale("log", base=2)
    ax.set_title(a.title); ax.set_xlabel(a.xlabel or a.x); ax.set_ylabel(a.ylabel)
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}")

if __name__ == "__main__": main()
```

- [ ] **Step 2: Write `scripts/gpu_pod.yaml`** (reusable single-GPU debug pod)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-debug
spec:
  restartPolicy: Never
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  containers:
  - name: cuda
    image: nvcr.io/nvidia/pytorch:24.10-py3   # CUDA + PyTorch + nsys + nccl-tests baseline
    command: ["sleep", "infinity"]
    resources:
      limits:
        nvidia.com/gpu: 1
    volumeMounts:
    - { name: dshm, mountPath: /dev/shm }
  volumes:
  - name: dshm
    emptyDir: { medium: Memory }
```

- [ ] **Step 3: Install local Python deps + validate plot helper**

Run: `pip3 install pandas matplotlib >/dev/null; printf 'x,y\n1,10\n2,20\n' > /tmp/t.csv && python3 scripts/lib_plot.py --csv /tmp/t.csv --x x --y y --out /tmp/t.png && ls -l /tmp/t.png`
Expected: `wrote /tmp/t.png` and a non-zero-size PNG.

- [ ] **Step 4: Validate the pod manifest server-side**

Run: `kubectl apply --dry-run=server -f scripts/gpu_pod.yaml`
Expected: `pod/gpu-debug created (server dry run)`.

- [ ] **Step 5: Commit**

```bash
git add scripts/lib_plot.py scripts/gpu_pod.yaml
git commit -m "scaffold: plotting helper and reusable GPU debug pod"
```

### Task 0.3: Guide overview doc + reference skeletons

**Files:**
- Create: `docs/00-guide-overview.md`
- Create: `reference/glossary.md`, `reference/xid-table.md`, `reference/nccl-tunables.md`, `reference/driver-matrix.md`, `reference/tool-cheatsheets.md`, `reference/reference-arch-cheatsheet.md`

- [ ] **Step 1: Write `docs/00-guide-overview.md`** with these sections (prose, no measured numbers yet): (a) What this guide is and the four-part journey; (b) Cluster-at-a-glance table (copy the §2 spec table); (c) The GCP machine-family + GCP↔NVIDIA product-mapping tables (copy from spec §1); (d) How to run the labs safely — never touch DWS holders, all changes additive, teardown per lab; (e) Reading order and the doc↔lab↔toolkit three-way spine.

- [ ] **Step 2: Write reference skeletons** — each file gets a title + one-paragraph purpose + a table header to be filled as later phases produce content. `xid-table.md`: columns `XID | Meaning | Typical cause | Action`. `nccl-tunables.md`: `Env var | Effect | When to set`. `driver-matrix.md`: copy the machine-type→driver table from the sibling `gcp-ai-infra-study` repo's `gpu_driver_troubleshooting_guide.md` §1.2 (re-verify against `nvidia-smi` output in Task 2.x). `tool-cheatsheets.md`: one section per tool with the 3–5 commands used in the labs. `reference-arch-cheatsheet.md`: DGX/HGX/SuperPOD one-line definitions. `glossary.md`: alphabetized terms (SM, SIMT, warp, HBM, NVLink, NVSwitch, RDMA, RoCE, GPUDirect, TCPX/TCPXO, NCCL, ring/tree, bus vs algo bandwidth, gang scheduling, DWS, DPU, SuperNIC, SHARP, rail-optimized, scalable unit, NVL72).

- [ ] **Step 3: Link-check** — verify relative links resolve.

Run: `grep -roE '\]\(([^)]+\.md)' docs/00-guide-overview.md | sed 's/](//' | while read f; do test -e "docs/$f" -o -e "$f" || echo "MISSING: $f"; done; echo done`
Expected: `done` with no `MISSING` lines (create stubs for any missing).

- [ ] **Step 4: Commit**

```bash
git add docs/00-guide-overview.md reference/
git commit -m "docs: guide overview and reference skeletons"
```

---

## Phase 1 — NVIDIA toolkit reference (cross-cutting)

These six docs are reference-grade explanations of the tools. They contain **command syntax and interpretation guidance** (safe to write without a run), but any **example output** embedded must come from a real run captured in a later lab (link forward, or embed after the relevant lab runs). Write the explanatory content now; backfill real example snippets as labs produce them.

### Task 1.1: `docs/toolkit/T1-monitoring-inventory.md`
Cover: `nvidia-smi` (default table, `-q`, `-q -d MEMORY,UTILIZATION,CLOCK,POWER,ECC,TEMPERATURE`, `dmon`, `pmon`, `topo -m`), NVML (what it is, Python `pynvml`), **DCGM** architecture (`nv-hostengine`, `dcgmi discovery -l`, `dcgmi dmon`, field groups), `dcgm-exporter` (metrics for Prometheus), `nvtop`/`gpustat`. For each: what it measures, when to use it, one exact command. Add "Portability" note: identical on any NVIDIA host; on GKE COS run inside a GPU pod. Commit.

### Task 1.2: `docs/toolkit/T2-health-diagnostics.md`
Cover: DCGM diagnostics levels (`dcgmi diag -r 1|2|3|4`, what each runs, expected runtime), `nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAP`, **XID** errors (what they are, where they appear — `dmesg`, `/var/log/syslog`, DCGM), thermal/power throttle reasons (`nvidia-smi -q -d PERFORMANCE` clocks-throttle-reasons bitmask), `nvidia-bug-report.sh`, `gpu-burn` (stress + verify). Symptom→cause→action framing. Cross-link `reference/xid-table.md`. Commit.

### Task 1.3: `docs/toolkit/T3-profiling-tracing.md`
Cover: **Nsight Systems** (`nsys profile -t cuda,nvtx,nccl -o out ./app`, timeline concepts, `nsys stats`), **Nsight Compute** (`ncu --set full -o out ./app`, sections, roofline, **requires elevated GPU perf-counter permission** — flag that GKE COS may restrict this; note the `NVreg_RestrictProfilingToAdminUsers` / `--cap-add SYS_ADMIN` requirement and document the observed limitation), CUPTI & NVTX (annotating code), PyTorch profiler (`torch.profiler` + `tensorboard`/Chrome trace), Holistic Trace Analysis (HTA) for multi-rank traces. Commit.

### Task 1.4: `docs/toolkit/T4-benchmarking.md`
Cover: methodology (warmup, repeats, reporting median, **bus bandwidth vs algorithm bandwidth** definitions), `nvbandwidth` (H2D/D2H/P2P tests), CUDA-samples `bandwidthTest`, **nccl-tests** (`all_reduce_perf`, `all_gather_perf`, `-b`/`-e`/`-f`/`-g`/`-n` flags, how it reports `busbw`/`algbw`), cuBLAS/cuDNN microbench pointers, `gpu-burn`. Commit.

### Task 1.5: `docs/toolkit/T5-networking-fabric-tools.md`
Cover: `NCCL_DEBUG=INFO`/`NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` (reading topology/ring/tree/transport lines), NCCL topology dump (`NCCL_TOPO_DUMP_FILE`), GPUDirect verification (how to confirm the transport in NCCL logs), `perftest` (`ib_write_bw`/`ib_send_bw` — for RDMA fabrics; note applicability to A3 Ultra/A4 not A3 High), `ethtool -S` NIC counters, RoCE/ECN counters, `ibstat`/`mlxlink` (Mellanox/CX-7 — flag: present on RDMA families, not A3 High gVNIC). Commit.

### Task 1.6: `docs/toolkit/T6-portability-matrix.md`
A matrix: rows = capability/tool (driver install, GPU inventory, health diag, profiling, intra-node fabric, inter-node fabric, collectives, scheduling, monitoring); columns = GCP A3/A4 · on-prem DGX/HGX · other clouds · Slurm · bare-metal K8s. Cells name the concrete mechanism per platform. Include the GCP↔NVIDIA product mapping from spec §1. Commit.

---

## Phase 2 — Part I: Single Node

### Task 2.1: Lab 01 + doc 01 — GPU microarchitecture (`lab-01-gpu-arch-inspect`, `docs/part1-single-node/01-gpu-microarchitecture.md`)

**Files:** Create `labs/lab-01-gpu-arch-inspect/{run.sh,README.md}`, `docs/part1-single-node/01-gpu-microarchitecture.md`.

- [ ] **Step 1: Write `labs/lab-01-gpu-arch-inspect/run.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib_capture.sh"
LAB=lab-01
kubectl apply -f "$REPO_ROOT/scripts/gpu_pod.yaml"
kubectl wait --for=condition=Ready pod/gpu-debug --timeout=300s
cap_run $LAB smi.txt          -- kubectl exec gpu-debug -- nvidia-smi
cap_run $LAB smi-q.txt        -- kubectl exec gpu-debug -- nvidia-smi -q
cap_run $LAB topo.txt         -- kubectl exec gpu-debug -- nvidia-smi topo -m
cap_run $LAB devquery.txt     -- kubectl exec gpu-debug -- bash -lc 'deviceQuery || /usr/local/cuda/extras/demo_suite/deviceQuery || echo "deviceQuery not present"'
cap_verify_provenance $LAB assets/$LAB "$(kubectl get pod gpu-debug -o jsonpath='{.spec.nodeName}')" "arch inspect"
```

- [ ] **Step 2: Validate:** `bash -n labs/lab-01-gpu-arch-inspect/run.sh && kubectl apply --dry-run=server -f scripts/gpu_pod.yaml` → dry-run OK.
- [ ] **Step 3: Execute:** `bash labs/lab-01-gpu-arch-inspect/run.sh` → pod Ready, four files written.
- [ ] **Step 4: Verify artifacts:** `for f in smi smi-q topo devquery; do test -s assets/lab-01/$f.txt || echo "EMPTY $f"; done; grep -q H100 assets/lab-01/smi.txt && echo OK` → `OK`, no `EMPTY`.
- [ ] **Step 5: Write `docs/part1-single-node/01-gpu-microarchitecture.md`** — sections: CPU vs GPU vs TPU (design philosophy); SM / SIMT / warps / branch divergence; memory hierarchy (registers→shared→L2→HBM3); Tensor Cores & FP8/BF16/TF32; **Hopper H100 specifics** (SM count, HBM3 capacity/bandwidth, TMA, thread-block clusters, transformer engine). Embed the **real** SM count / clock / memory / CC values from `assets/lab-01/smi-q.txt` in a "What our H100 actually reports" callout. End with "Practice → lab-01" and "Tools in this layer → T1". Attribute: note H100 = A3 family; contrast Blackwell (A4) in one line.
- [ ] **Step 6: Write `labs/lab-01-gpu-arch-inspect/README.md`** — objective, prereqs, the exact steps above, embedded excerpts from the four artifacts, interpretation (map `topo -m` legend), teardown (`kubectl delete pod gpu-debug`), links up to doc 01 + T1.
- [ ] **Step 7: Provenance already appended by run.sh; confirm a new row in `VERIFICATION.md`.**
- [ ] **Step 8: Commit** `git add labs/lab-01-gpu-arch-inspect docs/part1-single-node/01-gpu-microarchitecture.md VERIFICATION.md && git commit -m "part1: GPU microarchitecture doc + lab-01 (arch inspect)"`

### Task 2.2: Lab 02 + doc 02 — Drivers, CUDA & troubleshooting (`lab-02-driver-cuda-health`)

**Files:** Create `labs/lab-02-driver-cuda-health/{run.sh,README.md}`, `docs/part1-single-node/02-drivers-cuda-install-troubleshooting.md`; update `reference/xid-table.md`, `reference/driver-matrix.md`.

- [ ] **Step 1: Write `run.sh`** — capture: `nvidia-smi --query-gpu=driver_version,cuda_version,name,vbios_version --format=csv`; `nvidia-smi -q -d ECC,PERFORMANCE,TEMPERATURE,POWER`; node `dmesg | grep -i -E 'NVRM|Xid'` via a privileged node-debug pod (`kubectl debug node/<n>` with `nvidia/cuda` image) — if node dmesg is inaccessible on COS, capture `kubectl get events` + document the limitation; run DCGM diagnostics: `dcgmi diag -r 2` (install datacenter-gpu-manager in the pod or use `nvcr.io/nvidia/cloud-native/dcgm` image); run `gpu-burn` for 60s via image `oguzpastirmaci/gpu-burn` as a Job. Use `cap_run` for each; `cap_verify_provenance`.
- [ ] **Step 2: Validate** (`bash -n`, dry-run any Job manifest).
- [ ] **Step 3: Execute** on cluster.
- [ ] **Step 4: Verify** artifacts non-empty; `grep -qi 'Pass\|healthy' assets/lab-02/dcgm-diag.txt || echo "review diag"`.
- [ ] **Step 5: Write doc 02** — driver branches (LTSB/PB/NFB) + GCP matrix (from real `nvidia-smi` + sibling repo, re-verified); **GKE managed driver install** (device-plugin DaemonSets — reference the real `nvidia-gpu-device-plugin-large-cos` DS observed) **vs** GCE manual (`.run` installer) **vs generic NVIDIA GPU Operator**; CUDA toolkit/runtime vs driver compatibility; **XID taxonomy** (fill `reference/xid-table.md` with the common XIDs: 13,31,43,48,63,64,74,79,94,95 + meaning/cause/action); thermal/throttle reasons decode; the diagnostics workflow (smi → dcgmi diag → bug-report). Embed real driver/CUDA versions + gpu-burn result + dcgm-diag summary.
- [ ] **Step 6: Write lab README** (steps, embedded output, interpretation of XID/throttle, teardown).
- [ ] **Step 7: Commit.**

### Task 2.3: Lab 03 + doc 03 — Single-GPU execution & profiling (`lab-03-single-gpu-benchmark-profile`)

**Files:** Create `labs/lab-03-single-gpu-benchmark-profile/{run.sh,gemm.py,parse.py,README.md}`, `docs/part1-single-node/03-single-gpu-execution-and-profiling.md`.

- [ ] **Step 1: Write `gemm.py`** — a PyTorch script doing warmup + timed FP16/BF16/FP8 (if supported) square GEMMs across sizes `[2048,4096,8192,16384]`, printing `dtype,size,tflops` CSV lines; wrap the timed region in `torch.cuda.Event` and an NVTX range (`torch.cuda.nvtx.range`).
- [ ] **Step 2: Write `run.sh`** — on `gpu-debug` pod: (a) `nvbandwidth` (build from `github.com/NVIDIA/nvbandwidth` or use image; capture `nvbandwidth` H2D/D2H/P2P) → `assets/lab-03/nvbandwidth.txt`; (b) run `gemm.py` → `assets/lab-03/gemm.csv`; (c) `nsys profile -t cuda,nvtx -o /work/gemm python gemm.py` then `nsys stats /work/gemm.nsys-rep` → `assets/lab-03/nsys-stats.txt` (copy the `.nsys-rep` out via `kubectl cp`); (d) attempt `ncu --set full -o /work/gemm_ncu python gemm.py` — if it fails on COS due to perf-counter permission, capture the error and note the limitation. `cap_verify_provenance`.
- [ ] **Step 3: Write `parse.py`** — read `gemm.csv`, pivot dtype vs size, call `lib_plot.py` to produce `assets/lab-03/gemm_tflops.png` (bar, TFLOPs by size per dtype).
- [ ] **Step 4: Validate / Execute / Verify** (`gemm.csv` has rows; `gemm_tflops.png` non-empty; `nsys-stats.txt` contains kernel table).
- [ ] **Step 5: Write doc 03** — CUDA execution model (host→device, streams, kernels, occupancy), the roofline model, compute- vs memory-bound; how one training/inference step maps to kernels; interpret the **real** nsys timeline + measured TFLOPs vs H100 peak + nvbandwidth H2D/D2H/P2P numbers; roofline placement of the GEMM. Note ncu limitation honestly if hit.
- [ ] **Step 6: Write lab README** + embed `gemm_tflops.png`, nvbandwidth numbers, nsys stats excerpt.
- [ ] **Step 7: Commit.**

### Task 2.4: Lab 04 + doc 04 — Intra-node NVLink/NVSwitch (HGX) (`lab-04-intranode-nvlink-hgx`)

**Files:** Create `labs/lab-04-intranode-nvlink-hgx/{run.sh,parse.py,README.md}`, `docs/part1-single-node/04-intranode-nvlink-nvswitch-hgx.md`, plus a reusable `manifests/nccl-single-node.yaml` (8-GPU pod).

- [ ] **Step 1: Write `manifests/nccl-single-node.yaml`** — one pod requesting `nvidia.com/gpu: 8`, `/dev/shm` (medium Memory, sized ~16Gi), image `nvcr.io/nvidia/pytorch:24.10-py3` (ships nccl-tests) or build nccl-tests in an initContainer; command sleeps.
- [ ] **Step 2: Write `run.sh`** — capture `nvidia-smi topo -m` (full 8-GPU matrix) and `nvidia-smi nvlink -s` (per-link state/bandwidth); run `nvbandwidth` P2P matrix; run single-node `all_reduce_perf -b 8 -e 8G -f 2 -g 8` → `assets/lab-04/allreduce.txt`. `cap_verify_provenance`.
- [ ] **Step 3: Write `parse.py`** — parse `all_reduce_perf` output (size, algbw, busbw) → `assets/lab-04/allreduce.csv`; plot busbw vs size (`--logx`) → `assets/lab-04/allreduce_busbw.png`.
- [ ] **Step 4: Validate / Execute / Verify** (csv rows present; busbw approaches NVLink regime — record actual).
- [ ] **Step 5: Write doc 04** — the 8-GPU HGX H100 baseboard (NVLink gen4, NVSwitch, ~900 GB/s bisection), NVLink vs PCIe paths, **why intra-node ≫ inter-node** (sets up Parts II–IV); read the **real** topo matrix and single-node busbw curve; **explicitly frame the node as an HGX H100 baseboard** and forward-reference Part IV §11. Attribute: A3/A4 all use HGX-class baseboards.
- [ ] **Step 6: Write lab README** + embed topo matrix, nvlink state, busbw plot.
- [ ] **Step 7: Commit.**

**Phase 2 ships:** Part I is a complete, standalone single-node deep-dive with 4 runnable labs.

---

## Phase 3 — Part II: Inter-node Communication

### Task 3.1: Lab 05 + doc 05 — NICs, RDMA, GPUDirect (`lab-05-network-path-inspect`)

**Files:** Create `labs/lab-05-network-path-inspect/{run.sh,README.md}`, `docs/part2-inter-node/05-nic-rdma-gpudirect.md`.

- [ ] **Step 1: Write `run.sh`** — characterize the ACTUAL path: node NICs (`kubectl get node <n> -o jsonpath` for `networking.gke.io` annotations + `ip -br link` via node-debug pod), extended resources (`kubectl describe node | grep -A20 Allocatable`), installed network DaemonSets (`kubectl get ds -A | grep -iE 'tcpx|fastsocket|rdma|nccl'`), and a 2-pod inter-node bandwidth baseline over pod network (`iperf3` server/client pinned to the two GPU nodes via `nodeSelector`) → `assets/lab-05/`. `cap_verify_provenance`.
- [ ] **Step 2–4: Validate / Execute / Verify** (confirm the earlier finding: only `nvidia.com/gpu:8`, no GPU-NIC resource, fastsocket DS 0-scheduled; record iperf3 Gbps).
- [ ] **Step 5: Write doc 05** — gVNIC vs dedicated GPU NICs; RDMA/RoCE vs GPUDirect-TCPX (A3 High) vs TCPXO (A3 Mega) vs GPUDirect-RDMA (A3 Ultra/A4); the NCCL network-plugin DaemonSet model; `NCCL_*`/`NCCL_GPUDIRECTTCPX_*` env; **document this cluster's real path** (standard gVNIC/TCP, no GPUDirect) with the captured evidence. Portability + product-attribution table. Include an **optional** subsection: how one would deploy the Fast Socket / TCPX plugin and what extended resources would then appear (mark as not-enabled-here unless Step 6 succeeds).
- [ ] **Step 6 (optional, if feasible & reversible):** attempt to deploy the NCCL Fast Socket installer / GPUDirect-TCPX plugin per GKE docs; if it activates and exposes resources, capture the delta for use in lab-06 A/B. If not feasible on this cluster form, document why. Never disturb DWS holders.
- [ ] **Step 7: Write lab README + commit.**

### Task 3.2: Lab 06 + doc 06 — NCCL collectives, 2-node sweep (`lab-06-nccl-tests-internode`)

**Files:** Create `labs/lab-06-nccl-tests-internode/{mpijob.yaml,run.sh,parse.py,README.md}`, `docs/part2-inter-node/06-nccl-collectives.md`; install `mpi-operator`.

- [ ] **Step 1: Install Kubeflow `mpi-operator`** — `kubectl apply -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.6.0/deploy/v2beta1/mpi-operator.yaml`; verify `kubectl get crd mpijobs.kubeflow.org`.
- [ ] **Step 2: Write `manifests`→`mpijob.yaml`** — an MPIJob: 1 launcher + 2 workers (`nodeSelector` the DWS pool, `nvidia.com/gpu: 8` each, `/dev/shm`), image with nccl-tests + OpenMPI (`nvcr.io/nvidia/pytorch:24.10-py3`), launcher runs `mpirun -np 16 --hostfile ... all_reduce_perf -b 8 -e 8G -f 2 -g 1` with `NCCL_DEBUG=INFO`. Set `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH`.
- [ ] **Step 3: Write `run.sh`** — apply MPIJob, stream launcher logs → `assets/lab-06/allreduce-2node.txt` and `assets/lab-06/nccl-debug.txt`; repeat for `all_gather_perf`; run each 3× for stable medians; if the TCPX plugin was enabled in lab-05, run an A/B (standard vs plugin). `cap_verify_provenance`.
- [ ] **Step 4: Write `parse.py`** — parse busbw/algbw vs size for all-reduce & all-gather → CSVs; plot 2-node busbw curves (`assets/lab-06/busbw_curves.png`); if A/B exists, overlay.
- [ ] **Step 5: Validate / Execute / Verify** (MPIJob `Succeeded`; csv rows; the NCCL log shows the transport — record whether it's `Socket`/`NET/IB`/`NET/FastSocket`).
- [ ] **Step 6: Write doc 06** — all-reduce/all-gather/reduce-scatter mechanics; ring vs tree; topology discovery; message-size vs bandwidth; **bus vs algo bandwidth** derivation; interpret the **real** 2-node curves and the transport line from `nccl-debug.txt`; compare 2-node inter-node busbw to lab-04 single-node busbw (the intra ≫ inter point, quantified). Product attribution (what the curve would look like on TCPXO/RDMA families).
- [ ] **Step 7: Write lab README + commit.**

**Phase 3 ships:** the inter-node communication story, quantified and honest about the actual data path.

---

## Phase 4 — Part III: Clustering & Distributed Execution

### Task 4.1: Lab 07 + doc 07 — GKE scheduling & topology (`lab-07-gke-gang-schedule`)
- Manifest: a 2-replica indexed Job (or JobSet) each requesting `nvidia.com/gpu: 8`, demonstrating gang placement across both nodes; show `kubectl get pods -o wide` node spread. Doc: device plugin internals, requests/limits/tolerations, topology-aware & gang scheduling, DWS + capacity holders; generic equivalents (GPU Operator, NFD, Slurm gres). Capture real pod→node placement. Steps follow the standard LAB sequence; commit.

### Task 4.2: Lab 08 + doc 08 — Job frameworks: JobSet + Kueue (`lab-08-jobset-multinode`)
- Install JobSet controller + Kueue; create a ClusterQueue/LocalQueue with GPU quota; write `manifests/jobset-nccl.yaml` (replicated job, headless service, pod DNS rendezvous) running a 16-GPU workload. Doc: JobSet replicated-job model, Kueue queueing/quota/preemption, headless-service + pod-DNS rendezvous, why gang scheduling matters. Capture admission + pod DNS + completion. Commit.

### Task 4.3: Lab 09 + doc 09 — Distributed training DDP/FSDP + profiling (`lab-09-2node-pytorch-ddp-fsdp-profile`)
- Write `train.py` (toy transformer; DDP mode and FSDP mode via a flag; `torch.profiler` with NCCL activities; NVTX ranges). Manifest: JobSet with `torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d` across both nodes. `run.sh`: run DDP, run FSDP, capture per-rank PyTorch traces + `nsys` on rank 0, pull traces via `kubectl cp`; run HTA over the multi-rank traces → `assets/lab-09/hta_*.png/csv`. Doc: DDP vs FSDP (sharding, comm/compute overlap), gradient bucketing → all-reduce/reduce-scatter+all-gather mapping, `torchrun` rendezvous, scaling efficiency; interpret real traces (comm vs compute time, exposed comm) + HTA output; compare DDP vs FSDP measured step time. Commit.

### Task 4.4: Lab 10 + doc 10 — Fleet observability & debugging (`lab-10-observability-fleet-debug`)
- Deploy `dcgm-exporter` (DaemonSet, `nvcr.io/nvidia/k8s/dcgm-exporter`) + `kube-prometheus-stack` (Helm) with a Grafana GPU dashboard; capture a Grafana panel screenshot/JSON during a running job. Then **fault injection** (each reversible): (a) kill one rank mid-job → capture the NCCL timeout/abort signature; (b) saturate a NIC with iperf3 during nccl-tests → capture the busbw degradation; (c) force a version/env mismatch (`NCCL_ALGO` invalid or mismatched image) → capture the failure. Doc: the metrics pipeline (DCGM→exporter→Prometheus→Grafana), slow/stalled/mismatched collective diagnosis, XID at scale, hangs/timeouts; each injected fault as symptom→signature→root cause→fix. Commit. **Teardown documented; DWS holders untouched.**

**Phase 4 ships:** end-to-end distributed execution, profiling, and operations.

---

## Phase 5 — Part IV: Platform & Reference Architectures (knowledge-first)

Docs are reference-architecture explanations. **Banner on each:** "Knowledge-first — hardware not present on this cluster; see the observe-and-compare lab-11 for what IS measurable here." No fabricated hardware output.

### Task 5.1: doc 11 — DGX / HGX systems & troubleshooting
- HGX baseboard (what our node is) vs full DGX H100/H200 system (chassis, dual CPU, NVSwitch trays, storage/NICs); DGX OS, **NVSM** (`nvsm show health`), **GPU Fabric Manager** (role for NVSwitch, its logs/errors), Base Command Manager. System-level troubleshooting walkthrough (NVSM health, NVLink/NVSwitch diagnostics, BMC/sideband, fabric-manager failure modes) described against the DGX platform. Map to GCP: which parts GCP manages/hides for A3 tenants. Commit.

### Task 5.2: doc 12 — BlueField DPUs & DOCA
- DPU role (offload net/storage/security from host CPU), DOCA stack overview, BlueField-3 SuperNIC; where DPUs sit in an AI cluster (north-south vs east-west, isolation). Contrast: GCP **Titanium** offload plays the equivalent role on GCP (tenant doesn't manage a DPU). **Link out to the NVIDIA DOCA skills** (telemetry, DMS, Argus) for hands-on rather than duplicating. Commit.

### Task 5.3: doc 13 — Spectrum-X & AI fabrics
- Spectrum-X Ethernet (Spectrum-4 switch + BlueField-3 SuperNIC, adaptive routing, RoCE congestion control, packet spraying) vs **Quantum InfiniBand** (rail-optimized topology, **SHARP** in-network reduction) vs cloud fabric/GPUDirect-TCPX. Map each to what NCCL sees (transport, algo). Explicitly compare to lab-06's measured curves and explain what would change on Spectrum-X/IB. Product attribution: A3 Ultra/A4 RoCE ↔ Spectrum-X/IB class. Commit.

### Task 5.4: doc 14 — DGX SuperPOD
- Reference architecture: scalable unit (SU), compute/storage/management fabrics, rail-optimized IB, **NVLink Switch System / GB200 NVL72** scale-up domains, Base Command Manager / Run:ai orchestration, validation & acceptance testing. Contrast with how cloud A3/A4 scales (GKE + DWS + JobSet/Kueue). Commit.

### Task 5.5: Lab 11 — platform compare (`lab-11-platform-compare`)
- `run.sh`: map the A3 node to the HGX H100 baseboard (reuse lab-04 topo), probe for and **document the absence** of Fabric Manager (`nvidia-smi -q | grep -i fabric`; `nv-fabricmanager` not present), BlueField (`lspci | grep -i bluefield` → none), Spectrum-X SuperNIC (NIC model via ethtool → gVNIC). Build `assets/lab-11/a3-vs-dgx-superpod.csv` comparison and render a table. README banner: "runnable-here = the probes; read-only = the DGX/SuperPOD reference material." Doc-lab links. Commit.

**Phase 5 ships:** the platform capstone, clearly separating measured facts from reference knowledge.

---

## Phase 6 — Integration & polish

### Task 6.1: Cross-linking & spine integrity
- Ensure every doc has "Practice → lab" and "Tools in this layer → T#" sections and every lab links back. Run a link-checker over `docs/` and `labs/` (`grep`-based, as in Task 0.3) → no `MISSING`. Commit.

### Task 6.2: Backfill toolkit example outputs
- Insert real captured snippets (from labs) into `docs/toolkit/T1–T5` example blocks, replacing forward-references. Commit.

### Task 6.3: Finalize reference cheat-sheets
- Complete `reference/xid-table.md`, `nccl-tunables.md` (from NCCL_DEBUG output actually seen), `tool-cheatsheets.md`, `driver-matrix.md` (re-verified), `reference-arch-cheatsheet.md`. Commit.

### Task 6.4: README status flip + final read-through
- Update `README.md` status from "🚧 planning" to "✅ built"; verify the four-part TOC links resolve; confirm `VERIFICATION.md` has a row per runnable lab. Commit and push.

---

## Self-Review (completed by plan author)

**Spec coverage:** Purpose/journey → Phases 2–5. Toolkit layer → Phase 1 + Task 6.2. Product-mapping/portability → Task 0.3, T6, and every doc's attribution note. Live-lab data flow → `scripts/lib_capture.sh`/`lib_plot.py` + per-lab run.sh/parse.py. Verification principles → Global Constraints + VERIFICATION.md appended by every lab. Part IV honesty → Phase 5 banners + Task 5.5 absence-probes. Success criteria (single→cluster→platform, tools actually run, real data, injected failures, product attribution) → Phases 2–6.

**Placeholder scan:** No "TBD/TODO". Doc-writing steps specify exact sections and which real artifact to embed rather than inventing numbers (required by the execute-before-documenting constraint). Scripts/manifests are given as concrete code.

**Type/name consistency:** `cap_run`, `cap_nodes`, `cap_verify_provenance`, `cap_gpu_exec` defined in Task 0.1 and used consistently. `lib_plot.py` CLI flags consistent across parse.py callers. Artifact paths `assets/lab-XX/` consistent with `labs/lab-XX-*/` naming.

**Known execution risks flagged in-plan (not placeholders):** `ncu` perf-counter permission on COS (Task 1.3/2.3), node `dmesg`/XID access on COS (Task 2.2), and GPUDirect-TCPX enablement feasibility (Task 3.1 Step 6) — each has a documented fallback consistent with the honesty principle.
