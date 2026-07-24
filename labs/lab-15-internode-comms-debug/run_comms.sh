#!/usr/bin/env bash
# Lab 15 — inter-node comms debug: read what NCCL chose, then localize two
# comms-layer faults (Flex-safe, all faults injected at the job/env level).
#
# Runs a 24-GPU all-reduce across all 3 nodes inside a guarded GPU-borrow
# window and captures three things a healthy-path lab never shows:
#   1. HEALTHY BASELINE with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH —
#      the transport/algorithm/channels NCCL *actually chose* (Socket vs TCPX,
#      Ring vs Tree, #channels). This is the "read the wire" skill lab-10 lacks.
#   2. FAULT A — a WRONG NCCL_SOCKET_IFNAME (a non-existent interface): NCCL
#      can't find a usable NIC and fails at init/bootstrap. The INFO log names
#      the missing interface — that is how you localize an iface/routing fault.
#   3. FAULT B — a STRAGGLER (one rank sleeps past the PG timeout before the
#      first collective): the other 23 ranks block in all_reduce and the
#      watchdog aborts them at *exactly* the PG timeout — the hang signature,
#      distinct from a peer crash (which aborts in seconds via a closed socket).
#
# WHY 3 NODES: a comms fault is only interesting when there is a real inter-node
# fabric to misconfigure and multiple peers to strand. Using all 3 nodes / 24
# GPUs also means no node sits idle during the borrow window (always-hold rule).
#
# GPU SAFETY: identical guarded borrow window to lab-12/lab-13b — scale
# gpu-holder 3->0, occupy with 3 workbench pods, EXIT-trap restores the hold on
# every exit path. No node is drained or deleted (Flex-safe); every fault is a
# per-run env var or a job-level sleep, and each phase is bounded in wall-clock.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
OUT="${REPO_ROOT}/assets/lab-15"; mkdir -p "$OUT"
LAB06="${REPO_ROOT}/labs/lab-06-2node-nccl-collectives"
HERE="${REPO_ROOT}/labs/lab-15-internode-comms-debug"

PORT="${LAB15_PORT:-29541}"
IMAGE="${LAB15_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
PG_TIMEOUT="${LAB15_PG_TIMEOUT:-45}"     # bounds a blocked/straggling collective
STRAGGLER_SLEEP="${LAB15_STRAGGLER_SLEEP:-70}"  # > PG_TIMEOUT so survivors abort first
FAULT_WAIT="${LAB15_FAULT_WAIT:-75}"     # wall-clock bound per fault phase, then force-return
BAD_IFACE="${LAB15_BAD_IFACE:-nonexistent0}"
PODS=(nccl-wb-a nccl-wb-b nccl-wb-c)

mapfile -t NODES < <(cap_nodes | sort)
[ "${#NODES[@]}" -ge 3 ] || { echo "[lab-15] FATAL: need 3 nodes, found ${#NODES[@]}" >&2; exit 1; }
echo "[lab-15] nodes: ${NODES[*]}"

cleanup() {
  echo "[lab-15] cleanup: freeing workbenches and restoring gpu-holder to 3"
  kubectl delete pod "${PODS[@]}" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

echo "[lab-15] scaling gpu-holder 3->0"
kubectl scale deploy gpu-holder --replicas=0
kubectl wait --for=delete pod -l app=gpu-holder --timeout=180s || true

gen_workbench() {
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: $1
  labels: { app: nccl-workbench }
spec:
  hostNetwork: true
  nodeName: $2
  restartPolicy: Never
  tolerations:
  - operator: Exists
  containers:
  - name: bench
    image: ${IMAGE}
    command: ["sleep", "infinity"]
    securityContext:
      capabilities: { add: ["IPC_LOCK"] }
    resources:
      limits: { nvidia.com/gpu: "8" }
    volumeMounts:
    - { name: dshm, mountPath: /dev/shm }
  volumes:
  - name: dshm
    emptyDir: { medium: Memory, sizeLimit: 16Gi }
EOF
}

for i in 0 1 2; do gen_workbench "${PODS[$i]}" "${NODES[$i]}"; done
echo "[lab-15] waiting for workbench pods..."
kubectl wait --for=condition=Ready pod "${PODS[@]}" --timeout=300s
for P in "${PODS[@]}"; do
  kubectl cp "${HERE}/comms_bench.py"      "${P}:/workspace/comms_bench.py"
  kubectl cp "${LAB06}/launch_node.sh"     "${P}:/workspace/launch_node.sh"
done

master_ip="$(kubectl exec "${PODS[0]}" -- bash -lc 'hostname -I | awk "{print \$1}"')"
echo "[lab-15] master_ip=${master_ip}"

clear_logs() { for P in "${PODS[@]}"; do kubectl exec "$P" -- bash -lc 'rm -f /workspace/w_*.log' 2>/dev/null || true; done; }

# launch_phase <common-env-string> — background one launch per node, RANK 0 first.
launch_phase() {
  local envv="$1"; shift
  LAUNCH_PIDS=()
  for node_r in 0 1 2; do
    local pod="${PODS[$node_r]}"
    kubectl exec "$pod" -- bash -lc \
      "cd /workspace && NNODES=3 NODE_NAME=${NODES[$node_r]} BENCH_SCRIPT=/workspace/comms_bench.py ${envv} bash launch_node.sh ${node_r} ${master_ip} ${PORT} 8" &
    LAUNCH_PIDS+=($!)
    [ "$node_r" -eq 0 ] && sleep 6
  done
  return 0   # never let the trailing false test trip the inherited `set -e`
}

# =============================================================================
# Phase 1 — HEALTHY BASELINE: read what NCCL actually chose (transport/algo).
# =============================================================================
echo "[lab-15] === Phase 1: healthy baseline (NCCL_DEBUG=INFO INIT,NET,GRAPH) ==="
clear_logs
launch_phase "NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PG_TIMEOUT=${PG_TIMEOUT} COMMS_ITERS=20 COMMS_NBYTES=$((256*1024*1024))"
healthy_rc=0; for p in "${LAUNCH_PIDS[@]}"; do wait "$p" || healthy_rc=1; done
kubectl exec "${PODS[0]}" -- bash -lc 'cat /workspace/w_0.log' > "${OUT}/comms_healthy_rank0_full.txt" 2>/dev/null || true
# Distil the transport/algorithm/channel selection lines a triager reads first.
grep -Ei 'NET/|Using network|via NET|net plugin|NCCL INFO Bootstrap|Channel|Ring|Trees|nChannels|Connected all|comm .* rank .* nranks|COMMS all_reduce|# done' \
  "${OUT}/comms_healthy_rank0_full.txt" 2>/dev/null | head -n 40 > "${OUT}/comms_healthy_transport.txt" || true
echo "[lab-15] healthy rc=${healthy_rc}; transport summary:"
cat "${OUT}/comms_healthy_transport.txt" 2>/dev/null || true

# =============================================================================
# Phase 2 — FAULT A: wrong NCCL_SOCKET_IFNAME (non-existent interface).
# =============================================================================
echo "[lab-15] === Phase 2: FAULT A — NCCL_SOCKET_IFNAME=${BAD_IFACE} (init/bootstrap failure) ==="
clear_logs
launch_phase "NCCL_SOCKET_IFNAME=${BAD_IFACE} NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PG_TIMEOUT=${PG_TIMEOUT} COMMS_ITERS=5 COMMS_NBYTES=$((64*1024*1024))"
# Bound the phase: the failure is usually fast, but cap wall-clock and force-return.
( sleep "$FAULT_WAIT"; for P in "${PODS[@]}"; do kubectl exec "$P" -- bash -lc 'pkill -9 -f comms_bench.py' 2>/dev/null || true; done ) &
GUARD=$!
for p in "${LAUNCH_PIDS[@]}"; do wait "$p" || true; done
kill "$GUARD" 2>/dev/null || true; wait "$GUARD" 2>/dev/null || true
kubectl exec "${PODS[0]}" -- bash -lc 'cat /workspace/w_0.log' > "${OUT}/comms_fault_badiface_rank0.txt" 2>/dev/null || true
grep -Ei "${BAD_IFACE}|No interface found|no socket interface|NET/|WARN|error|Bootstrap|retrieveNetwork|abort" \
  "${OUT}/comms_fault_badiface_rank0.txt" 2>/dev/null | head -n 25 > "${OUT}/comms_fault_badiface_signature.txt" || true
echo "[lab-15] FAULT A signature:"
cat "${OUT}/comms_fault_badiface_signature.txt" 2>/dev/null || true

# =============================================================================
# Phase 3 — FAULT B: straggler (one rank arrives after the PG timeout).
# =============================================================================
echo "[lab-15] === Phase 3: FAULT B — straggler rank 16 sleeps ${STRAGGLER_SLEEP}s (hang to PG timeout) ==="
clear_logs
launch_phase "NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PG_TIMEOUT=${PG_TIMEOUT} COMMS_ITERS=20 COMMS_NBYTES=$((256*1024*1024)) STRAGGLER_RANK=16 STRAGGLER_SLEEP=${STRAGGLER_SLEEP}"
# Bound the phase: survivors abort at PG_TIMEOUT; the straggler wakes later and
# then fails too. Cap wall-clock, capture, and force-return so the window closes.
( sleep $((PG_TIMEOUT + FAULT_WAIT)); for P in "${PODS[@]}"; do kubectl exec "$P" -- bash -lc 'pkill -9 -f comms_bench.py' 2>/dev/null || true; done ) &
GUARD=$!
for p in "${LAUNCH_PIDS[@]}"; do wait "$p" || true; done
kill "$GUARD" 2>/dev/null || true; wait "$GUARD" 2>/dev/null || true
# rank 0 (survivor on node 0) sees the hang; rank 16 (straggler on node 2) is the culprit.
kubectl exec "${PODS[0]}" -- bash -lc 'cat /workspace/w_0.log'  > "${OUT}/comms_fault_straggler_rank0.txt"  2>/dev/null || true
kubectl exec "${PODS[2]}" -- bash -lc 'cat /workspace/w_16.log' > "${OUT}/comms_fault_straggler_rank16.txt" 2>/dev/null || true
grep -Ei 'ARRIVE|STRAGGLER|iter=|Timeout|timed out|watchdog|abort|WorkNCCL|DistBackend|NCCL' \
  "${OUT}/comms_fault_straggler_rank0.txt" 2>/dev/null | head -n 25 > "${OUT}/comms_fault_straggler_signature.txt" || true
echo "[lab-15] FAULT B signature (survivor rank 0):"
cat "${OUT}/comms_fault_straggler_signature.txt" 2>/dev/null || true
clear_logs

# =============================================================================
# Timeline summary
# =============================================================================
{
  echo "# lab-15 inter-node comms debug — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# cluster asia-east1-c  nodes: ${NODES[0]} ${NODES[1]} ${NODES[2]}"
  echo "# PG_TIMEOUT=${PG_TIMEOUT}s  straggler_sleep=${STRAGGLER_SLEEP}s  bad_iface=${BAD_IFACE}"
  echo ""
  echo "## Phase 1 — healthy: transport/algorithm NCCL chose (rank 0):"
  cat "${OUT}/comms_healthy_transport.txt" 2>/dev/null | head -n 20 || true
  echo ""
  echo "## Phase 2 — FAULT A (wrong NCCL_SOCKET_IFNAME=${BAD_IFACE}) signature:"
  cat "${OUT}/comms_fault_badiface_signature.txt" 2>/dev/null | head -n 12 || true
  echo ""
  echo "## Phase 3 — FAULT B (straggler rank 16) survivor signature:"
  cat "${OUT}/comms_fault_straggler_signature.txt" 2>/dev/null | head -n 12 || true
} > "${OUT}/comms_timeline.txt"

echo "[lab-15] ============ timeline summary ============"
cat "${OUT}/comms_timeline.txt"

cap_verify_provenance "lab-15" "assets/lab-15" "${NODES[0]},${NODES[1]},${NODES[2]}" \
  "asia-east1-c inter-node comms triage: healthy NCCL_DEBUG=INFO transport read + wrong-NCCL_SOCKET_IFNAME init fault + straggler hang-to-PG-timeout"
echo "[lab-15] done. (holder restored by EXIT trap)"
