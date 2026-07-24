#!/usr/bin/env bash
# Lab 14 — single-GPU & node health triage (Flex-safe, single-node borrow).
#
# Reads a GPU's health the way you do on day 2: not "is it up" but "is it
# healthy *under load*, and if it's slow, WHY." Six captures, all on one
# borrowed node inside a guarded window:
#   A. IDLE health baseline    — nvidia-smi -q -d ECC,PERFORMANCE,CLOCK,POWER,
#      TEMPERATURE,ROW_REMAP,PCIE : the reference every fault is read against.
#   B. UNDER-LOAD delta        — 8-GPU sustained matmul + `nvidia-smi dmon`
#      (power/clocks/util/VIOLATIONS/temp) and a re-read of the PERFORMANCE
#      group: clocks at boost, power near TDP, throttle reasons still-or-not.
#   C. DETERMINISTIC throttle  — temporarily cap device-0 power limit (min),
#      load it, and read a REAL "SW Power Cap" throttle reason going *Active*
#      with a measured clock drop. Fully reversed afterwards (and by the trap).
#   D. dcgmi diag -r 3         — the long, deployment-grade suite (lab-02 ran -r 2).
#      NOTE: no stock public DCGM diag image ships cuda13 plugins for this node's
#      R580/CUDA-13 driver yet, so -r 3 could NOT run here (gotchas G12/G13); the
#      captured output is the honest version-matching failure, not a passing diag.
#   E. PRIVILEGED ncu rerun    — the half of profiling lab-03 could NOT finish:
#      with CAP_SYS_ADMIN, `ncu --set full` now collects real per-kernel metrics
#      instead of ERR_NVGPUCTRPERM.
#   F. XID GKE-native surface  — DCGM_FI_DEV_XID_ERRORS off dcgm-exporter, the
#      privileged-container dmesg read, and the NPD->Cloud Logging path.
#
# WHY THIS LAB (vs the healthy-path labs 01-03): those confirmed the GPU exists
# and benchmarked it once. This one reads health *under sustained load*, induces
# and reads a *real throttle reason*, and completes the privileged profiling that
# lab-03 was blocked from doing. Single-GPU health does not need 3 nodes — so it
# borrows just ONE node (holder 3->2), keeping the other two held (always-hold).
#
# GPU SAFETY / Flex-safe: scale gpu-holder 3->2 (frees exactly one node's 8 GPUs),
# occupy that node with one 8-GPU workbench pod, EXIT-trap restores the power
# limit AND the holder to 3 on every path. No node drain/cordon/delete. The only
# device-state change is a power-limit cap that is explicitly restored (Phase C).
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
set +e   # this lab runs several intentionally-failing / time-bounded commands; manage errors explicitly (see gotcha G1)

LAB="lab-14"
OUT="${ASSETS}/${LAB}"; mkdir -p "$OUT"
LAB03="${REPO_ROOT}/labs/lab-03-single-gpu-benchmark-profile"
HERE="${REPO_ROOT}/labs/lab-14-single-gpu-health-triage"

POD="${LAB14_POD:-gpu-health-wb}"
IMAGE="${LAB14_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
LOAD_SECONDS="${LAB14_LOAD_SECONDS:-90}"
CAP_LOAD_SECONDS="${LAB14_CAP_LOAD_SECONDS:-45}"

mapfile -t NODES < <(cap_nodes | sort)
[ "${#NODES[@]}" -ge 3 ] || { echo "[lab-14] FATAL: need the 3-node pool, found ${#NODES[@]}" >&2; exit 1; }
echo "[lab-14] pool nodes: ${NODES[*]}"

# --- state the cleanup trap must undo -----------------------------------------
DEF_PL=""          # device-0 default power limit (W), captured before capping
PL_CHANGED=0       # set to 1 once we cap the power limit

cleanup() {
  echo "[lab-14] cleanup: restore power limit, free workbench+dcgm-job, restore gpu-holder=3"
  if [ "$PL_CHANGED" = "1" ] && [ -n "$DEF_PL" ]; then
    kubectl exec "$POD" -- nvidia-smi -i 0 -pl "$DEF_PL" 2>/dev/null || true
  fi
  kubectl delete pod "$POD" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl delete job gpu-dcgm-diag --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

# =============================================================================
# Borrow ONE node: holder 3->2, then find the node with no holder pod on it.
# =============================================================================
echo "[lab-14] scaling gpu-holder 3->2 (frees one node's 8 GPUs)"
kubectl scale deploy gpu-holder --replicas=2
echo "[lab-14] waiting for gpu-holder to settle at 2 pods..."
for _ in $(seq 1 60); do
  n=$(kubectl get pods -l app=gpu-holder -o name 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = "2" ] && break
  sleep 3
done
held="$(kubectl get pods -l app=gpu-holder -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u)"
FREE_NODE=""
for node in "${NODES[@]}"; do grep -qx "$node" <<<"$held" || { FREE_NODE="$node"; break; }; done
[ -n "$FREE_NODE" ] || { echo "[lab-14] FATAL: could not identify a freed node (held=[$held])" >&2; exit 1; }
echo "[lab-14] borrowing free node: $FREE_NODE"

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
  labels: { app: gpu-health-wb }
spec:
  nodeName: ${FREE_NODE}
  restartPolicy: Never
  tolerations:
  - operator: Exists
  containers:
  - name: bench
    image: ${IMAGE}
    command: ["sleep", "infinity"]
    securityContext:
      privileged: true          # CAP_SYS_ADMIN for ncu perf counters + power-limit + dmesg
    resources:
      limits: { nvidia.com/gpu: "7" }   # 7 GPUs; the 8th is left for the DCGM diag Job (node stays fully held)
    volumeMounts:
    - { name: dshm, mountPath: /dev/shm }
  volumes:
  - name: dshm
    emptyDir: { medium: Memory, sizeLimit: 16Gi }
EOF

echo "[lab-14] waiting for workbench pod..."
kubectl wait --for=condition=Ready "pod/${POD}" --timeout=300s || { echo "[lab-14] FATAL: pod not ready" >&2; exit 1; }
kubectl cp "${HERE}/load_gpu.py" "${POD}:/workspace/load_gpu.py"
kubectl cp "${LAB03}/gemm.py"    "${POD}:/workspace/gemm.py"

# DCGM diag Job on the 8th GPU of the SAME node — runs concurrently so the whole
# node stays held (no idle GPUs) while the workbench drives Phases A/B/C/E/F.
echo "[lab-14] launching dcgmi diag -r 3 as a 1-GPU Job on ${FREE_NODE} (concurrent)"
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata: { name: gpu-dcgm-diag }
spec:
  backoffLimit: 0
  template:
    spec:
      nodeName: ${FREE_NODE}
      restartPolicy: Never
      tolerations: [{ operator: Exists }]
      containers:
      - name: dcgm
        # DCGM 4.4.x matches the node's R580 driver (managed dcgm-exporter is 4.4.1);
        # the older 3.3.8 image fails "Detected unsupported Cuda version" (gotcha G13).
        image: ${LAB14_DCGM_IMAGE:-nvcr.io/nvidia/cloud-native/dcgm:4.4.0-1-ubuntu22.04}
        command: ["/bin/bash","-lc"]
        args: ["nv-hostengine >/dev/null 2>&1; sleep 3; dcgmi diag -r 3"]
        resources: { limits: { nvidia.com/gpu: "1" } }
EOF

# =============================================================================
# Phase A — IDLE health baseline (device 0). The reference read.
# =============================================================================
echo "[lab-14] === Phase A: idle health baseline (device 0) ==="
cap_run "$LAB" "health_idle_full.txt" -- \
  kubectl exec "$POD" -- nvidia-smi -i 0 -q -d ECC,PERFORMANCE,CLOCK,POWER,TEMPERATURE,ROW_REMAPPER >/dev/null
grep -Ei 'Product Name|GPU Current Temp|Power Draw|Current Power Limit|SM |Graphics|Clocks Throttle|SW Power Cap|HW Slowdown|HW Thermal|SW Thermal|Sync Boost|Idle|Applications Clocks|Correctable|Uncorrectable|Remapped|Pending|Failure Occurred|Row Remap' \
  "${OUT}/health_idle_full.txt" 2>/dev/null | head -n 60 > "${OUT}/health_idle_signature.txt"
echo "[lab-14] idle signature:"; sed -n '1,30p' "${OUT}/health_idle_signature.txt"

# =============================================================================
# Phase B — UNDER LOAD: 8-GPU matmul + dmon + PERFORMANCE re-read (the delta).
# =============================================================================
echo "[lab-14] === Phase B: under-load telemetry delta (all 8 GPUs) ==="
kubectl exec "$POD" -- bash -lc \
  "for d in 0 1 2 3 4 5 6; do CUDA_VISIBLE_DEVICES=\$d LOAD_SECONDS=${LOAD_SECONDS} MATMUL_N=8192 python3 /workspace/load_gpu.py >/workspace/load_\$d.log 2>&1 & done; wait" &
LOAD_PID=$!
sleep 20   # let clocks/power/temp ramp to steady state
echo "[lab-14] sampling nvidia-smi dmon under load (power/clocks/util/violations/temp)..."
cap_run "$LAB" "dmon_under_load.txt" -- \
  kubectl exec "$POD" -- nvidia-smi dmon -s pucvmet -c 20 >/dev/null
cap_run "$LAB" "health_load_full.txt" -- \
  kubectl exec "$POD" -- nvidia-smi -i 0 -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE >/dev/null
grep -Ei 'GPU Current Temp|Power Draw|Current Power Limit|SM |Graphics|Clocks Throttle|SW Power Cap|HW Slowdown|HW Thermal|SW Thermal|Sync Boost|Idle|HW Power Brake' \
  "${OUT}/health_load_full.txt" 2>/dev/null | head -n 40 > "${OUT}/health_load_signature.txt"
echo "[lab-14] under-load signature:"; sed -n '1,25p' "${OUT}/health_load_signature.txt"
wait "$LOAD_PID" 2>/dev/null

# =============================================================================
# Phase C — DETERMINISTIC, REVERSIBLE throttle: cap device-0 power, load it,
# read a REAL "SW Power Cap" throttle reason going Active. Then restore.
# =============================================================================
echo "[lab-14] === Phase C: deterministic power-cap throttle (device 0, reversible) ==="
DEF_PL="$(kubectl exec "$POD" -- bash -lc "nvidia-smi -i 0 --query-gpu=power.default_limit --format=csv,noheader,nounits" 2>/dev/null | tr -d ' \r' | cut -d. -f1)"
MIN_PL="$(kubectl exec "$POD" -- bash -lc "nvidia-smi -i 0 --query-gpu=power.min_limit --format=csv,noheader,nounits" 2>/dev/null | tr -d ' \r' | cut -d. -f1)"
echo "[lab-14] device-0 power: default=${DEF_PL}W min=${MIN_PL}W"
if [ -n "$DEF_PL" ] && [ -n "$MIN_PL" ] && kubectl exec "$POD" -- nvidia-smi -i 0 -pl "$MIN_PL" >/tmp/lab14_pl 2>&1; then
  PL_CHANGED=1
  echo "[lab-14] capped device-0 power limit to ${MIN_PL}W; loading..."
  kubectl exec "$POD" -- bash -lc \
    "CUDA_VISIBLE_DEVICES=0 LOAD_SECONDS=${CAP_LOAD_SECONDS} MATMUL_N=8192 python3 /workspace/load_gpu.py >/workspace/load_cap.log 2>&1 &"
  sleep 15
  cap_run "$LAB" "throttle_powercap_full.txt" -- \
    kubectl exec "$POD" -- nvidia-smi -i 0 -q -d PERFORMANCE,CLOCK,POWER >/dev/null
  grep -Ei 'Power Draw|Power Limit|SM |Graphics|Clocks Throttle|SW Power Cap|HW Slowdown|HW Thermal|SW Thermal|Sync Boost|Idle|HW Power Brake' \
    "${OUT}/throttle_powercap_full.txt" 2>/dev/null | head -n 40 > "${OUT}/throttle_powercap_signature.txt"
  echo "[lab-14] power-cap throttle signature:"; sed -n '1,25p' "${OUT}/throttle_powercap_signature.txt"
  kubectl exec "$POD" -- bash -lc 'pkill -9 -f load_gpu.py' 2>/dev/null
  echo "[lab-14] restoring device-0 power limit to ${DEF_PL}W"
  kubectl exec "$POD" -- nvidia-smi -i 0 -pl "$DEF_PL" 2>/dev/null
  PL_CHANGED=0
  kubectl exec "$POD" -- bash -lc "nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader" \
    | tee "${OUT}/throttle_powercap_restored.txt"
else
  echo "[lab-14] power-limit change not permitted on this device; skipping Phase C (see note)" \
    | tee "${OUT}/throttle_powercap_signature.txt"
  cat /tmp/lab14_pl 2>/dev/null | tee -a "${OUT}/throttle_powercap_signature.txt"
fi

# =============================================================================
# Phase D — harvest dcgmi diag -r 3 (the concurrent DCGM Job). lab-02 ran -r 2.
# =============================================================================
echo "[lab-14] === Phase D: dcgmi diag -r 3 (deployment-grade, via DCGM Job) ==="
echo "[lab-14] waiting for gpu-dcgm-diag Job to reach a terminal state (bounded)..."
# Break as soon as EITHER Complete or Failed is true (don't block 720s on a failed job).
d_end=$((SECONDS + 800))
while [ "$SECONDS" -lt "$d_end" ]; do
  cond="$(kubectl get job gpu-dcgm-diag -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)"
  [ -n "$cond" ] && { echo "[lab-14] dcgm job terminal: $cond"; break; }
  # also bail out fast on an image-pull failure
  pw="$(kubectl get pods -l job-name=gpu-dcgm-diag -o jsonpath='{.items[*].status.containerStatuses[*].state.waiting.reason}' 2>/dev/null)"
  case "$pw" in *ImagePull*|*ErrImage*) echo "[lab-14] dcgm job image pull failing: $pw"; break;; esac
  sleep 6
done
DCGM_IMG_USED="${LAB14_DCGM_IMAGE:-nvcr.io/nvidia/cloud-native/dcgm:4.4.0-1-ubuntu22.04}"
{ echo "# cmd: dcgmi diag -r 3   (DCGM image ${DCGM_IMG_USED}, 1 GPU on ${FREE_NODE})";
  kubectl logs job/gpu-dcgm-diag 2>&1; } > "${OUT}/dcgm_diag_r3.txt"
echo "[lab-14] dcgm diag -r 3 tail:"; tail -n 18 "${OUT}/dcgm_diag_r3.txt" 2>/dev/null

# =============================================================================
# Phase E — PRIVILEGED ncu rerun: the half lab-03 could NOT finish.
# =============================================================================
echo "[lab-14] === Phase E: privileged ncu rerun (completes lab-03) ==="
kubectl exec "$POD" -- bash -lc \
  'cd /workspace && timeout 360 ncu --set full --target-processes all --launch-count 3 --launch-skip 20 -f -o /workspace/gemm_ncu python3 gemm.py' \
  > "${OUT}/ncu_privileged_full.txt" 2>&1
echo "# --- proof: no ERR_NVGPUCTRPERM, and a real metrics page ---" > "${OUT}/ncu_privileged_metrics.txt"
kubectl exec "$POD" -- bash -lc \
  'ncu -i /workspace/gemm_ncu.ncu-rep --page details 2>/dev/null | grep -iE "Kernel Name|Compute \(SM\)|Memory Throughput|DRAM Throughput|Duration|Achieved Occupancy|SM Frequency|ncclDevKernel|ampere|sm90|cutlass|gemm|elementwise" | head -n 40' \
  >> "${OUT}/ncu_privileged_metrics.txt" 2>&1
echo "[lab-14] ncu tail (should show profiling, not ERR_NVGPUCTRPERM):"; tail -n 12 "${OUT}/ncu_privileged_full.txt"
echo "[lab-14] ncu metrics proof:"; sed -n '1,20p' "${OUT}/ncu_privileged_metrics.txt"

# =============================================================================
# Phase F — XID GKE-native surface (DCGM metric + privileged dmesg + NPD path).
# =============================================================================
echo "[lab-14] === Phase F: XID GKE-native surface ==="
# F1: privileged container CAN read dmesg (contrast lab-02's unprivileged pod).
kubectl exec "$POD" -- bash -lc 'dmesg 2>&1 | grep -iE "NVRM|Xid" | tail -n 30 || echo "(no NVRM/Xid lines — healthy)"' \
  | tee "${OUT}/dmesg_nvrm.txt"
# F2: what the MANAGED dcgm-exporter actually exports — and whether XID is in it.
DCGM_POD="$(kubectl get pods -n gke-managed-system -o name 2>/dev/null | grep dcgm-exporter | head -1 | cut -d/ -f2)"
if [ -n "$DCGM_POD" ]; then
  echo "[lab-14] scraping managed dcgm-exporter ${DCGM_POD} for XID field"
  kubectl port-forward -n gke-managed-system "pod/${DCGM_POD}" 9400:9400 >/dev/null 2>&1 &
  PF_PID=$!; sleep 4
  {
    echo "# from ${DCGM_POD} — does the MANAGED exporter expose an XID field?"
    xid="$(curl -s localhost:9400/metrics 2>/dev/null | grep -iE 'DCGM_FI_DEV_XID' | head -n 5)"
    if [ -n "$xid" ]; then echo "$xid";
    else echo "# NONE — DCGM_FI_DEV_XID_ERRORS is NOT in the managed exporter's default field set."; fi
    echo "# --- fields the managed exporter DOES export: ---"
    curl -s localhost:9400/metrics 2>/dev/null | grep -oE '^DCGM_FI_[A-Z_]+' | sort -u
  } | tee "${OUT}/dcgm_xid_metric.txt"
  kill "$PF_PID" 2>/dev/null; wait "$PF_PID" 2>/dev/null
else
  echo "# dcgm-exporter pod not found in gke-managed-system" | tee "${OUT}/dcgm_xid_metric.txt"
fi
# F3: NPD-raised node events (Xid conditions surface here, not in-container dmesg).
kubectl get events -A --field-selector "involvedObject.name=${FREE_NODE}" 2>/dev/null \
  | grep -iE 'xid|nvrm|gpu|Warning' | head -n 15 | tee "${OUT}/node_events_xid.txt"
[ -s "${OUT}/node_events_xid.txt" ] || echo "(no GPU/Xid node events — healthy node)" | tee "${OUT}/node_events_xid.txt"

# =============================================================================
# Timeline summary + provenance
# =============================================================================
{
  echo "# lab-14 single-GPU & node health triage — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# cluster asia-east1-c   borrowed node: ${FREE_NODE}   (holder held the other 2)"
  echo "# device-0 power: default=${DEF_PL}W min=${MIN_PL}W"
  echo ""
  echo "## A. idle health signature:"; sed -n '1,18p' "${OUT}/health_idle_signature.txt" 2>/dev/null
  echo ""
  echo "## B. under-load signature (device 0):"; sed -n '1,12p' "${OUT}/health_load_signature.txt" 2>/dev/null
  echo ""
  echo "## C. power-cap throttle signature (device 0):"; sed -n '1,14p' "${OUT}/throttle_powercap_signature.txt" 2>/dev/null
  echo ""
  echo "## E. privileged ncu (tail):"; tail -n 8 "${OUT}/ncu_privileged_full.txt" 2>/dev/null
  echo ""
  echo "## F. XID surface (DCGM metric):"; sed -n '1,8p' "${OUT}/dcgm_xid_metric.txt" 2>/dev/null
} > "${OUT}/health_timeline.txt"
echo "[lab-14] ============ timeline summary ============"; cat "${OUT}/health_timeline.txt"

cap_verify_provenance "lab-14" "assets/lab-14" "${FREE_NODE}" \
  "asia-east1-c single-GPU/node health: idle+under-load telemetry delta, reversible power-cap throttle (SW Power Cap Active), dcgmi diag -r 3, PRIVILEGED ncu rerun (completes lab-03), XID via DCGM_FI_DEV_XID_ERRORS + privileged dmesg"
echo "[lab-14] done. (power limit + holder restored by EXIT trap)"
