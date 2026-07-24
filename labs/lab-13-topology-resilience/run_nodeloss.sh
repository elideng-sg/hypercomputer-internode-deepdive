#!/usr/bin/env bash
# Lab 13b — job-level node-loss resilience & the 16-GPU survivor set (Flex-safe).
#
# Runs a 24-GPU all-reduce loop across all 3 nodes, then KILLS the ranks on ONE
# node mid-run (SIGKILL to that node's python processes — a job-level fault, NOT
# a node drain/delete, so Flex capacity is never released). It captures:
#   1. the FAULT SIGNATURE the surviving 16 ranks emit when a peer vanishes
#      (NCCL transport error + watchdog abort), made fast/diagnosable via a
#      bounded PG timeout + async error handling instead of a silent hang;
#   2. the VICTIM node's log (heartbeats that simply stop at the kill);
#   3. a SURVIVOR-SET RERUN — a fresh 16-GPU / 2-node all-reduce on the two
#      surviving nodes that runs to completion.
#
# WHY 3 NODES: lose 1 of 2 and the remnant is a single non-distributed node —
# there is no surviving *multi-node* set to observe or reschedule onto. Only at
# N>=3 does killing a node leave a genuine distributed survivor set. This is the
# resilience story two nodes physically cannot tell.
#
# GPU SAFETY: identical guarded borrow window to lab-12 — scale gpu-holder 3->0,
# occupy with 3 workbench pods, EXIT-trap restores the hold on every exit path.
# No node is drained or deleted (Flex-safe); only job processes are killed.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
OUT="${REPO_ROOT}/assets/lab-13"; mkdir -p "$OUT"
LAB06="${REPO_ROOT}/labs/lab-06-2node-nccl-collectives"
HERE="${REPO_ROOT}/labs/lab-13-topology-resilience"

PORT="${LAB13_PORT:-29531}"          # 24-GPU node-loss job
PORT2="${LAB13_PORT2:-29532}"        # 16-GPU survivor-set rerun
IMAGE="${LAB13_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
PG_TIMEOUT="${LAB13_PG_TIMEOUT:-90}" # bounds how long survivors block on a dead peer
SETTLE="${LAB13_SETTLE:-30}"         # seconds of healthy running before the kill
PODS=(nccl-wb-a nccl-wb-b nccl-wb-c)
VICTIM_IDX="${LAB13_VICTIM_IDX:-2}"  # kill the ranks on the 3rd node

mapfile -t NODES < <(cap_nodes | sort)
[ "${#NODES[@]}" -ge 3 ] || { echo "[lab-13b] FATAL: need 3 nodes, found ${#NODES[@]}" >&2; exit 1; }
echo "[lab-13b] nodes: ${NODES[*]}  (victim = ${NODES[$VICTIM_IDX]})"

cleanup() {
  echo "[lab-13b] cleanup: freeing workbenches and restoring gpu-holder to 3"
  kubectl delete pod "${PODS[@]}" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

echo "[lab-13b] scaling gpu-holder 3->0"
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
echo "[lab-13b] waiting for workbench pods..."
kubectl wait --for=condition=Ready pod "${PODS[@]}" --timeout=300s
for P in "${PODS[@]}"; do
  kubectl cp "${HERE}/nodeloss_bench.py"      "${P}:/workspace/nodeloss_bench.py"
  kubectl cp "${LAB06}/allreduce_bench.py"    "${P}:/workspace/allreduce_bench.py"
  kubectl cp "${LAB06}/launch_node.sh"        "${P}:/workspace/launch_node.sh"
done

master_ip="$(kubectl exec "${PODS[0]}" -- bash -lc 'hostname -I | awk "{print \$1}"')"
echo "[lab-13b] master_ip=${master_ip}"

# --- Phase 1: launch the 24-GPU node-loss job across all 3 nodes -------------
echo "[lab-13b] === launching 24-GPU all-reduce loop (3 nodes) ==="
LAUNCH_PIDS=()
for node_r in 0 1 2; do
  pod="${PODS[$node_r]}"
  envv="NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PG_TIMEOUT=${PG_TIMEOUT} NODE_NAME=${NODES[$node_r]} BENCH_SCRIPT=/workspace/nodeloss_bench.py"
  kubectl exec "$pod" -- bash -lc \
    "cd /workspace && NNODES=3 ${envv} bash launch_node.sh ${node_r} ${master_ip} ${PORT} 8" &
  LAUNCH_PIDS+=($!)
  [ "$node_r" -eq 0 ] && sleep 6
done

# --- Phase 2: let it run healthy, then kill the victim node's ranks ----------
echo "[lab-13b] job launched; letting it run healthy for ${SETTLE}s before the kill"
sleep "$SETTLE"
echo "[lab-13b] === KILLING ranks on victim node ${NODES[$VICTIM_IDX]} (pod ${PODS[$VICTIM_IDX]}) ==="
kubectl exec "${PODS[$VICTIM_IDX]}" -- bash -lc 'pkill -9 -f nodeloss_bench.py' || true

# --- Phase 3: wait for the launches to return (survivors fault, victim dies) -
echo "[lab-13b] waiting for ranks to exit (survivors fault within ~PG_TIMEOUT=${PG_TIMEOUT}s)..."
for p in "${LAUNCH_PIDS[@]}"; do wait "$p" || true; done

# --- Phase 4: collect the fault signature + victim log -----------------------
# rank 0 lives on node 0 (a survivor); rank VICTIM_IDX*8 is the victim's local-0.
SURV_POD="${PODS[0]}"
VICTIM_RANK=$((VICTIM_IDX * 8))
kubectl exec "$SURV_POD" -- bash -lc 'cat /workspace/w_0.log' \
  > "${OUT}/nodeloss_fault_survivor_rank0.txt" 2>/dev/null || true
kubectl exec "${PODS[$VICTIM_IDX]}" -- bash -lc "cat /workspace/w_${VICTIM_RANK}.log 2>/dev/null" \
  > "${OUT}/nodeloss_victim_rank${VICTIM_RANK}.txt" 2>/dev/null || true

echo "[lab-13b] --- survivor rank-0 tail (fault signature) ---"
tail -n 20 "${OUT}/nodeloss_fault_survivor_rank0.txt" 2>/dev/null || true

# Clear the job logs on the surviving pods before the rerun.
for P in "${PODS[0]}" "${PODS[1]}"; do
  kubectl exec "$P" -- bash -lc 'rm -f /workspace/w_*.log' 2>/dev/null || true
done

# --- Phase 5: survivor-set rerun — a real 16-GPU / 2-node job that completes -
echo "[lab-13b] === SURVIVOR-SET RERUN: 16-GPU all-reduce on nodes ${NODES[0]} + ${NODES[1]} ==="
SURV_PIDS=()
for node_r in 0 1; do
  pod="${PODS[$node_r]}"
  envv="NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN BENCH_MAX_EXP=28 BENCH_ITERS=10"
  kubectl exec "$pod" -- bash -lc \
    "cd /workspace && NNODES=2 ${envv} bash launch_node.sh ${node_r} ${master_ip} ${PORT2} 8" &
  SURV_PIDS+=($!)
  [ "$node_r" -eq 0 ] && sleep 6
done
survivor_rc=0; for p in "${SURV_PIDS[@]}"; do wait "$p" || survivor_rc=1; done
kubectl exec "${PODS[0]}" -- bash -lc 'cat /workspace/w_0.log' \
  > "${OUT}/survivor_set_rerun_16gpu.txt" 2>/dev/null || true
kubectl exec "${PODS[0]}" -- bash -lc 'rm -f /workspace/w_*.log' 2>/dev/null || true
echo "[lab-13b] survivor-set rerun rc=${survivor_rc}"

# --- Phase 6: distil a plain-text timeline summary ---------------------------
{
  echo "# lab-13b node-loss resilience — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# cluster asia-east1-c  nodes: ${NODES[0]} ${NODES[1]} ${NODES[2]}"
  echo "# victim node (ranks killed): ${NODES[$VICTIM_IDX]} (ranks ${VICTIM_RANK}-$((VICTIM_RANK+7)))"
  echo ""
  echo "## survivor fault signature (grep of survivor rank-0 log):"
  grep -Ei 'FAULT|error|abort|timeout|remote|closed|watchdog|NCCL' \
    "${OUT}/nodeloss_fault_survivor_rank0.txt" 2>/dev/null | head -n 25 || true
  echo ""
  echo "## victim rank-${VICTIM_RANK} last lines (heartbeats stop at the kill):"
  tail -n 5 "${OUT}/nodeloss_victim_rank${VICTIM_RANK}.txt" 2>/dev/null || true
  echo ""
  echo "## survivor-set rerun (16-GPU / 2-node) tail:"
  tail -n 8 "${OUT}/survivor_set_rerun_16gpu.txt" 2>/dev/null || true
  echo ""
  echo "## survivor-set rerun completed cleanly: $([ "$survivor_rc" -eq 0 ] && echo YES || echo 'NO (rc='"$survivor_rc"')')"
} > "${OUT}/nodeloss_timeline.txt"

echo "[lab-13b] ============ timeline summary ============"
cat "${OUT}/nodeloss_timeline.txt"

cap_verify_provenance "lab-13" "assets/lab-13" "${NODES[0]},${NODES[1]},${NODES[2]}" \
  "asia-east1-c job-level node-loss: killed ranks on ${NODES[$VICTIM_IDX]}, captured survivor fault + 16-GPU survivor-set rerun"
echo "[lab-13b] done. (holder restored by EXIT trap)"
