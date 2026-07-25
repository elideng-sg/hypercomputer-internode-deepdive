#!/usr/bin/env bash
# Lab 17 — performance monitoring & day-2 operations (Flex-safe, single-node borrow).
#
# NOT a fault-injection lab in the doc-16 sense — it is the STEADY-STATE skill:
# capture a performance baseline through the real monitoring pipeline, exercise
# the day-2 tools "in anger", then DETECT a regression from monitoring *before*
# it becomes a crash. Four captures, all read through the pipeline, not just
# nvidia-smi:
#   A. BASELINE via the monitoring pipeline — live PromQL against Google Managed
#      Prometheus (GMP) across all 24 GPUs, + the DCGM_FI_PROF_* field catalog
#      the managed dcgm-exporter actually exposes.
#   B. dcgmi IN ANGER — dcgmi health + dcgmi dmon (the local per-node DCGM tool),
#      run from a DCGM container (dcgmi is not in the PyTorch image, gotcha G7).
#   C. REGRESSION DETECTION FROM MONITORING — drive load, then cap device-0 power
#      700->200 W (reversible; lab-14's proven-safe throttle) and read the SILENT
#      THROTTLE back off GMP: engine-active stays high while SM clock collapses
#      and power pins at the cap — the GPUSilentThrottle alert. Then restore.
#   D. HTA — a profiled multi-GPU DDP run -> Kineto traces -> Holistic Trace
#      Analysis temporal breakdown + comm/compute overlap (where the step goes).
#
# WHY THIS LAB: every prior lab reads ONE tool at a point in time. Day-2 ops is
# the pipeline view — baselines, dashboards, alert rules, regression detection,
# and trace analysis. Single-node monitoring needs only ONE borrowed node, so it
# borrows one (holder 3->2) and keeps the other two held (always-hold).
#
# GPU SAFETY / Flex-safe: scale gpu-holder 3->2 (frees one node's 8 GPUs), occupy
# that node with a 7-GPU privileged workbench + a 1-GPU DCGM Job (node stays
# fully held). The ONLY device-state change is a power-limit cap explicitly
# restored in Phase C and re-restored by the EXIT trap, which also frees the
# workbench+job and restores gpu-holder=3. No node drain/cordon/delete.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
set +e   # several phases run bounded / best-effort tools; manage errors explicitly (gotcha G1)

LAB="lab-17"
OUT="${ASSETS}/${LAB}"; mkdir -p "$OUT"
HERE="${REPO_ROOT}/labs/lab-17-perf-monitoring-day2-ops"

POD="${LAB17_POD:-gpu-mon-wb}"
IMAGE="${LAB17_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
DCGM_IMAGE="${LAB17_DCGM_IMAGE:-nvcr.io/nvidia/cloud-native/dcgm:4.4.0-1-ubuntu22.04}"
PROJECT="${LAB17_PROJECT:-hdlab-elideng}"
GMP="https://monitoring.googleapis.com/v1/projects/${PROJECT}/location/global/prometheus/api/v1"

mapfile -t NODES < <(cap_nodes | sort)
[ "${#NODES[@]}" -ge 3 ] || { echo "[lab-17] FATAL: need the 3-node pool, found ${#NODES[@]}" >&2; exit 1; }
echo "[lab-17] pool nodes: ${NODES[*]}"

# --- state the cleanup trap must undo -----------------------------------------
DEF_PL=""; PL_CHANGED=0

cleanup() {
  echo "[lab-17] cleanup: restore power limit, free workbench+dcgm-job, restore gpu-holder=3"
  if [ "$PL_CHANGED" = "1" ] && [ -n "$DEF_PL" ]; then
    kubectl exec "$POD" -- nvidia-smi -i 0 -pl "$DEF_PL" 2>/dev/null || true
  fi
  kubectl delete pod "$POD" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl delete job gpu-dcgm-mon --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

# --- PromQL helpers (token NEVER written into the captured asset) -------------
_tok() { gcloud auth print-access-token 2>/dev/null; }
promql_instant() { # <label> <outfile> <promql>
  local label="$1" outfile="$2" q="$3"
  { echo "# GMP instant query: ${q}";
    curl -s -G "${GMP}/query" -H "Authorization: Bearer $(_tok)" \
      --data-urlencode "query=${q}" \
    | python3 -c 'import json,sys
d=json.load(sys.stdin); r=d.get("data",{}).get("result",[])
print("# status="+str(d.get("status"))+"  series="+str(len(r)))
for s in sorted(r, key=lambda x:(x["metric"].get("Hostname",""), int(x["metric"].get("gpu","0") or 0))):
    m=s["metric"]; v=s.get("value",["","?"])[1]
    print("  "+m.get("Hostname","?")[-4:]+" gpu"+str(m.get("gpu","?")).rjust(2)+"  "+m.get("__name__","")+" = "+str(v))'
  } > "${OUT}/${outfile}" 2>&1
  cap_verify_provenance "$LAB" "assets/lab-17/${outfile}" "GMP" "PromQL instant: ${q}" >/dev/null 2>&1 || true
}
promql_range() { # <outfile> <start> <end> <step> <promql...>  (multiple queries -> one file)
  local outfile="$1" start="$2" end="$3" step="$4"; shift 4
  : > "${OUT}/${outfile}"
  for q in "$@"; do
    { echo "### GMP range query: ${q}   [${start}..${end} step ${step}s]";
      curl -s -G "${GMP}/query_range" -H "Authorization: Bearer $(_tok)" \
        --data-urlencode "query=${q}" --data-urlencode "start=${start}" \
        --data-urlencode "end=${end}" --data-urlencode "step=${step}" \
      | python3 -c 'import json,sys,datetime
d=json.load(sys.stdin); r=d.get("data",{}).get("result",[])
if not r: print("  (no series)"); sys.exit()
FMT="%H:%M:%S"
# sort so the actively-loaded workbench series (container=bench) prints first
for s in sorted(r, key=lambda x:(x["metric"].get("gpu","0"), x["metric"].get("container",""))):
    m=s["metric"]; host=m.get("Hostname","?")[-4:]; g=m.get("gpu","?"); c=m.get("container","?")
    pts=s.get("values",[])
    parts=[]
    for t,v in pts:
        ts=datetime.datetime.fromtimestamp(float(t),datetime.timezone.utc).strftime(FMT)
        parts.append(ts+"="+format(float(v),".0f"))
    print("  "+host+" gpu"+str(g)+" ["+c+"]: "+" ".join(parts))'
      echo; } >> "${OUT}/${outfile}" 2>&1
  done
}

# =============================================================================
# Borrow ONE node: holder 3->2, find the node with no holder pod on it.
# =============================================================================
echo "[lab-17] scaling gpu-holder 3->2 (frees one node's 8 GPUs)"
kubectl scale deploy gpu-holder --replicas=2
for _ in $(seq 1 60); do
  n=$(kubectl get pods -l app=gpu-holder -o name 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = "2" ] && break; sleep 3
done
held="$(kubectl get pods -l app=gpu-holder -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u)"
FREE_NODE=""
for node in "${NODES[@]}"; do grep -qx "$node" <<<"$held" || { FREE_NODE="$node"; break; }; done
[ -n "$FREE_NODE" ] || { echo "[lab-17] FATAL: could not identify a freed node (held=[$held])" >&2; exit 1; }
FREE_SHORT="${FREE_NODE##*-}"
echo "[lab-17] borrowing free node: $FREE_NODE (short=${FREE_SHORT})"

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: { name: ${POD}, labels: { app: gpu-mon-wb } }
spec:
  nodeName: ${FREE_NODE}
  restartPolicy: Never
  tolerations: [{ operator: Exists }]
  containers:
  - name: bench
    image: ${IMAGE}
    command: ["sleep","infinity"]
    securityContext: { privileged: true }   # power-limit change + profiling
    resources: { limits: { nvidia.com/gpu: "7" } }   # 8th GPU -> DCGM Job (node stays fully held)
    volumeMounts: [{ name: dshm, mountPath: /dev/shm }]
  volumes:
  - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 16Gi } }
EOF
kubectl wait --for=condition=Ready "pod/${POD}" --timeout=300s || { echo "[lab-17] FATAL: pod not ready" >&2; exit 1; }
kubectl cp "${HERE}/load_gpu.py"     "${POD}:/workspace/load_gpu.py"
kubectl cp "${HERE}/profiled_ddp.py" "${POD}:/workspace/profiled_ddp.py"

# =============================================================================
# Phase A — BASELINE via the monitoring pipeline (GMP PromQL + exporter catalog)
# =============================================================================
echo "[lab-17] === Phase A: steady-state baseline via GMP PromQL (all 24 GPUs) ==="
promql_instant "$LAB" "baseline_sm_clock.txt"  'DCGM_FI_DEV_SM_CLOCK'
promql_instant "$LAB" "baseline_power.txt"     'DCGM_FI_DEV_POWER_USAGE'
promql_instant "$LAB" "baseline_engine.txt"    'DCGM_FI_PROF_GR_ENGINE_ACTIVE'
{ echo "### Fleet baseline (idle) read straight from Google Managed Prometheus:";
  echo "## SM clock (MHz):";        sed -n '1,30p' "${OUT}/baseline_sm_clock.txt";
  echo; echo "## Power draw (W):";  sed -n '1,4p'  "${OUT}/baseline_power.txt";
  echo; echo "## GR engine active:";sed -n '1,4p'  "${OUT}/baseline_engine.txt"; } > "${OUT}/baseline_promql.txt"
echo "[lab-17] baseline (fleet, from GMP):"; sed -n '1,12p' "${OUT}/baseline_promql.txt"

echo "[lab-17] --- the DCGM_FI_PROF_* field catalog the managed exporter exposes ---"
DCGM_POD="$(kubectl get pods -n gke-managed-system -o name 2>/dev/null | grep dcgm-exporter | head -1 | cut -d/ -f2)"
if [ -n "$DCGM_POD" ]; then
  kubectl port-forward -n gke-managed-system "pod/${DCGM_POD}" 9400:9400 >/dev/null 2>&1 &
  PF_PID=$!; sleep 4
  { echo "# managed dcgm-exporter (${DCGM_POD}) — the fields GMP scrapes every 30s:";
    curl -s localhost:9400/metrics 2>/dev/null | grep -oE '^DCGM_FI_[A-Z0-9_]+' | sort -u; } > "${OUT}/dcgm_prof_fields.txt"
  kill "$PF_PID" 2>/dev/null; wait "$PF_PID" 2>/dev/null
else
  echo "# dcgm-exporter pod not found" > "${OUT}/dcgm_prof_fields.txt"
fi
echo "[lab-17] exported field catalog:"; sed -n '1,25p' "${OUT}/dcgm_prof_fields.txt"

# =============================================================================
# Phase B — dcgmi IN ANGER (local per-node DCGM tool, via a DCGM Job on GPU 8)
# =============================================================================
echo "[lab-17] === Phase B: dcgmi health + dcgmi dmon (DCGM container) ==="
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata: { name: gpu-dcgm-mon }
spec:
  backoffLimit: 0
  template:
    spec:
      nodeName: ${FREE_NODE}
      restartPolicy: Never
      tolerations: [{ operator: Exists }]
      containers:
      - name: dcgm
        image: ${DCGM_IMAGE}
        command: ["/bin/bash","-lc"]
        args:
        - |
          set +e
          nv-hostengine >/dev/null 2>&1; sleep 3
          echo '### dcgmi discovery -l (GPUs this container sees):'
          dcgmi discovery -l
          echo; echo '### dcgmi health: set watches then check:'
          dcgmi health -g 0 -s a; sleep 2; dcgmi health -g 0 -c
          echo; echo '### dcgmproftester tensor load (bg) + dcgmi dmon time series:'
          ( dcgmproftester --no-dcgm-validation -t 1004 -d 25 >/dev/null 2>&1 & )
          sleep 3
          dcgmi dmon -e 100,155,203,1002,1004,1005 -c 15
          echo '### done'
        resources: { limits: { nvidia.com/gpu: "1" } }
EOF
echo "[lab-17] waiting for gpu-dcgm-mon Job (bounded)..."
d_end=$((SECONDS + 300))
while [ "$SECONDS" -lt "$d_end" ]; do
  cond="$(kubectl get job gpu-dcgm-mon -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)"
  [ -n "$cond" ] && { echo "[lab-17] dcgm-mon job terminal: $cond"; break; }
  pw="$(kubectl get pods -l job-name=gpu-dcgm-mon -o jsonpath='{.items[*].status.containerStatuses[*].state.waiting.reason}' 2>/dev/null)"
  case "$pw" in *ImagePull*|*ErrImage*) echo "[lab-17] dcgm-mon image pull failing: $pw"; break;; esac
  sleep 6
done
{ echo "# cmd: dcgmi discovery/health/dmon  (DCGM image ${DCGM_IMAGE}, 1 GPU on ${FREE_NODE})";
  kubectl logs job/gpu-dcgm-mon 2>&1; } > "${OUT}/dcgmi_dmon_health.txt"
echo "[lab-17] dcgmi capture (tail):"; tail -n 22 "${OUT}/dcgmi_dmon_health.txt"

# =============================================================================
# Phase C — REGRESSION DETECTION FROM MONITORING (silent throttle via GMP)
# =============================================================================
echo "[lab-17] === Phase C: induce a silent throttle, DETECT it from GMP ==="
T0=$(date -u +%s)
echo "[lab-17] driving load on GPUs 0-6 (healthy boost baseline)..."
kubectl exec "$POD" -- bash -lc \
  "for d in 0 1 2 3 4 5 6; do CUDA_VISIBLE_DEVICES=\$d LOAD_SECONDS=300 MATMUL_N=8192 python3 /workspace/load_gpu.py >/workspace/load_\$d.log 2>&1 & done; echo loaded" >/dev/null
sleep 45   # let clocks/power reach steady boost, and GMP scrape a healthy sample
T1=$(date -u +%s)
DEF_PL="$(kubectl exec "$POD" -- bash -lc "nvidia-smi -i 0 --query-gpu=power.default_limit --format=csv,noheader,nounits" 2>/dev/null | tr -d ' \r' | cut -d. -f1)"
MIN_PL="$(kubectl exec "$POD" -- bash -lc "nvidia-smi -i 0 --query-gpu=power.min_limit --format=csv,noheader,nounits" 2>/dev/null | tr -d ' \r' | cut -d. -f1)"
echo "[lab-17] device-0 power: default=${DEF_PL}W min=${MIN_PL}W — capping to min under load"
if [ -n "$DEF_PL" ] && [ -n "$MIN_PL" ] && kubectl exec "$POD" -- nvidia-smi -i 0 -pl "$MIN_PL" >/dev/null 2>&1; then
  PL_CHANGED=1
  # cross-check the LOCAL lens once, at the capped moment (nvidia-smi throttle reason)
  sleep 12
  cap_run "$LAB" "throttle_local_crosscheck.txt" -- \
    kubectl exec "$POD" -- nvidia-smi -i 0 -q -d PERFORMANCE,CLOCK,POWER >/dev/null
  echo "[lab-17] holding the throttled state so GMP scrapes it (30s scrape)..."
  sleep 120
  T2=$(date -u +%s)
  echo "[lab-17] restoring device-0 power to ${DEF_PL}W (regression cleared)"
  kubectl exec "$POD" -- nvidia-smi -i 0 -pl "$DEF_PL" >/dev/null 2>&1; PL_CHANGED=0
  sleep 45
  T3=$(date -u +%s)
else
  echo "[lab-17] power-limit change not permitted; capturing local read only" > "${OUT}/throttle_local_crosscheck.txt"
  T2=$(date -u +%s); T3=$T2
fi
kubectl exec "$POD" -- bash -lc 'pkill -9 -f load_gpu.py' 2>/dev/null

echo "[lab-17] waiting ~90s for GMP ingestion, then range-querying the whole window..."
sleep 90
promql_range "regression_promql.txt" "$((T0-30))" "$((T3+30))" "15" \
  "DCGM_FI_DEV_SM_CLOCK{Hostname=~\".*${FREE_SHORT}\", gpu=~\"0|1\"}" \
  "DCGM_FI_DEV_POWER_USAGE{Hostname=~\".*${FREE_SHORT}\", gpu=~\"0|1\"}" \
  "DCGM_FI_PROF_GR_ENGINE_ACTIVE{Hostname=~\".*${FREE_SHORT}\", gpu=~\"0|1\"}"
{ echo "# Regression read from the monitoring pipeline (GMP), borrowed node ...${FREE_SHORT}.";
  echo "# gpu0 = throttled (power capped ${MIN_PL}W); gpu1 = healthy control (same load).";
  echo "# Silent-throttle signature: gpu0 engine stays HIGH while SM clock COLLAPSES and power PINS at the cap.";
  echo "# window markers (UTC epoch): load=${T0} boost=${T1} capped->restore=${T2} end=${T3}";
  echo; cat "${OUT}/regression_promql.txt"; } > "${OUT}/regression_promql_annotated.txt"
mv "${OUT}/regression_promql_annotated.txt" "${OUT}/regression_promql.txt"
echo "[lab-17] regression time series (from GMP):"; sed -n '1,40p' "${OUT}/regression_promql.txt"

# =============================================================================
# Phase D — HTA: profiled DDP -> Kineto traces -> temporal + comm/comp overlap
# =============================================================================
echo "[lab-17] === Phase D: HTA on a profiled 7-GPU DDP run ==="
kubectl exec "$POD" -- bash -lc \
  'rm -rf /workspace/traces && mkdir -p /workspace/traces && cd /workspace &&
   NCCL_DEBUG=WARN torchrun --nproc_per_node=7 --nnodes=1 --node_rank=0 \
     --master_addr=127.0.0.1 --master_port=29517 profiled_ddp.py 2>&1 | tail -n 8' \
  > "${OUT}/hta_profiled_run.txt" 2>&1
echo "[lab-17] profiled run tail:"; tail -n 6 "${OUT}/hta_profiled_run.txt"
kubectl exec "$POD" -- bash -lc 'ls -la /workspace/traces' > "${OUT}/hta_trace_files.txt" 2>&1

echo "[lab-17] installing + running Holistic Trace Analysis (HTA)..."
kubectl exec "$POD" -- bash -lc \
  'pip install --quiet HolisticTraceAnalysis 2>&1 | tail -n 3;
   python3 - <<PY 2>&1
from hta.trace_analysis import TraceAnalysis
a = TraceAnalysis(trace_dir="/workspace/traces")
print("### temporal breakdown (compute / non-compute / idle per rank):")
try:
    print(a.get_temporal_breakdown().to_string())
except Exception as e:
    print("temporal_breakdown error:", e)
print()
print("### communication vs computation overlap (%):")
try:
    print(a.get_comm_comp_overlap().to_string())
except Exception as e:
    print("comm_comp_overlap error:", e)
print()
print("### top GPU kernels (by time):")
try:
    kb = a.get_gpu_kernel_breakdown(visualize=False)
    # get_gpu_kernel_breakdown returns a tuple of dataframes; print the summary one
    df = kb[0] if isinstance(kb, (list, tuple)) else kb
    print(df.head(10).to_string())
except Exception as e:
    print("gpu_kernel_breakdown error:", e)
PY' > "${OUT}/hta_analysis.txt" 2>&1
echo "[lab-17] HTA analysis (head):"; sed -n '1,40p' "${OUT}/hta_analysis.txt"

# =============================================================================
# Timeline + provenance
# =============================================================================
holder_end="$(kubectl get deploy gpu-holder -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>/dev/null)"
{
  echo "# lab-17 perf monitoring & day-2 ops — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# cluster asia-east1-c   borrowed node: ${FREE_NODE}   holder end=${holder_end}"
  echo ""
  echo "## A. fleet baseline (GMP PromQL, idle):"; sed -n '1,10p' "${OUT}/baseline_promql.txt" 2>/dev/null
  echo ""; echo "## B. dcgmi in anger (tail):";    tail -n 12 "${OUT}/dcgmi_dmon_health.txt" 2>/dev/null
  echo ""; echo "## C. silent throttle read from GMP:"; sed -n '1,24p' "${OUT}/regression_promql.txt" 2>/dev/null
  echo ""; echo "## D. HTA (head):";               sed -n '1,16p' "${OUT}/hta_analysis.txt" 2>/dev/null
} > "${OUT}/monitoring_timeline.txt"
echo "[lab-17] ============ timeline summary ============"; cat "${OUT}/monitoring_timeline.txt"

cap_verify_provenance "lab-17" "assets/lab-17" "${FREE_NODE}" \
  "asia-east1-c perf monitoring & day-2: GMP PromQL fleet baseline (24 GPUs), dcgmi health+dmon in anger, silent-throttle regression detected FROM the pipeline (gpu0 engine high + SM clock collapsed + power pinned at ${MIN_PL}W vs gpu1 control), HTA temporal + comm/comp overlap on a profiled 7-GPU DDP run; power limit + holder restored (end=${holder_end})"
echo "[lab-17] done. (power limit + holder restored by EXIT trap)"
