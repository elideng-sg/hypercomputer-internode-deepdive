#!/usr/bin/env bash
# Lab 13a — 24-GPU / 3-node gang via JobSet + Kueue (non-power-of-2 placement).
#
# Demonstrates what only >=3 nodes can show:
#   1. Kueue admits a 24-GPU JobSet as ONE Workload (gang, all-or-nothing).
#   2. The three 8-GPU pods are placed one-per-node across ALL THREE nodes.
#   3. The gang runs a real 24-rank NCCL all-reduce to completion (value=24.0).
#   4. A 32-GPU JobSet (4 replicas) is gang-GATED — QuotaReserved=False, zero
#      pods created — because it exceeds the 24-GPU ClusterQueue quota.
#
# Requires the JobSet + Kueue controllers (installed on asia-east1-c 2026-07-24).
#
# GPU SAFETY: same guarded borrow window as lab-12/13b — scale gpu-holder 3->0
# so the 24 GPUs are free for the gang, EXIT-trap deletes the JobSet/queues and
# restores the holder to 3 on every exit path. No node is drained (Flex-safe);
# the gang pods ARE the occupancy during the window.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
OUT="${REPO_ROOT}/assets/lab-13"; mkdir -p "$OUT"
M="${REPO_ROOT}/manifests"

QUEUES="${M}/kueue-gpu-queues-24.yaml"
JOBSET="${M}/jobset-nccl-24.yaml"
OVERQUOTA="$(mktemp "${CLAUDE_JOB_DIR:-/tmp}/tmp/jobset-overquota-32.XXXX.yaml" 2>/dev/null || mktemp)"
# Over-quota variant: rename + bump replicas 3->4 (=32 GPU > 24 quota).
sed -e 's/name: nccl-gang/name: nccl-gang-overquota/' \
    -e 's/replicas: 3 /replicas: 4 /' "$JOBSET" > "$OVERQUOTA"

mapfile -t NODES < <(cap_nodes | sort)
[ "${#NODES[@]}" -ge 3 ] || { echo "[lab-13a] FATAL: need 3 nodes, found ${#NODES[@]}" >&2; exit 1; }
echo "[lab-13a] nodes: ${NODES[*]}"

cleanup() {
  echo "[lab-13a] cleanup: deleting gang/over-quota JobSets + queues, restoring gpu-holder to 3"
  kubectl delete -f "$OVERQUOTA" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl delete -f "$JOBSET"    --wait=false --ignore-not-found 2>/dev/null || true
  kubectl delete -f "$QUEUES"    --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
  rm -f "$OVERQUOTA" 2>/dev/null || true
}
trap cleanup EXIT

echo "[lab-13a] scaling gpu-holder 3->0 (free 24 GPUs for the gang)"
kubectl scale deploy gpu-holder --replicas=0
kubectl wait --for=delete pod -l app=gpu-holder --timeout=180s || true

# --- 0. Controllers + quota objects -----------------------------------------
cap_run lab-13 gang_controllers.txt -- bash -c "echo '### JobSet + Kueue controllers'; kubectl --context ${KUBE_CONTEXT} get deploy -n jobset-system jobset-controller-manager -o wide; kubectl --context ${KUBE_CONTEXT} get deploy -n kueue-system kueue-controller-manager -o wide"
kubectl apply -f "$QUEUES"
sleep 3
cap_run lab-13 gang_queues.txt -- bash -c "echo '### ClusterQueue + LocalQueue (24-GPU quota)'; kubectl --context ${KUBE_CONTEXT} get clusterqueue gpu-cq-24 -o wide; kubectl --context ${KUBE_CONTEXT} get localqueue gpu-lq-24 -n default -o wide"

# --- 1. Submit the 24-GPU gang ----------------------------------------------
echo "[lab-13a] submitting 24-GPU gang JobSet"
kubectl apply -f "$JOBSET"
sleep 10
cap_run lab-13 gang_admission.txt -- bash -c "echo '### Kueue Workload for the gang (expect Admitted / QuotaReserved=True, 24 GPU)'; kubectl --context ${KUBE_CONTEXT} get workloads -n default -o wide; echo; kubectl --context ${KUBE_CONTEXT} get workloads -n default -o json | python3 -c 'import json,sys
for w in json.load(sys.stdin)[\"items\"]:
    n=w[\"metadata\"][\"name\"]; conds=[(c[\"type\"],c[\"status\"]) for c in w.get(\"status\",{}).get(\"conditions\",[])]
    adm=w.get(\"status\",{}).get(\"admission\",{}).get(\"clusterQueue\",\"-\")
    print(f\"  {n}: cq={adm} conds={conds}\")'"

# --- 2. Placement: one 8-GPU pod per node across all three nodes -------------
# Wait for pods to schedule.
for i in $(seq 1 60); do
  ready=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=nccl-gang -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | grep -c . || true)
  [ "${ready:-0}" -ge 3 ] && break; sleep 3
done
cap_run lab-13 gang_placement.txt -- bash -c "echo '### gang pods -> node placement (expect 3 pods, 3 distinct nodes, 8 GPU each)'; kubectl --context ${KUBE_CONTEXT} get pods -l jobset.sigs.k8s.io/jobset-name=nccl-gang -o wide; echo; echo '### distinct nodes used:'; kubectl --context ${KUBE_CONTEXT} get pods -l jobset.sigs.k8s.io/jobset-name=nccl-gang -o jsonpath='{range .items[*]}{.spec.nodeName}{\"\n\"}{end}' | sort -u"

# --- 3. Wait for completion, capture the 24-rank all-reduce result -----------
# Poll for terminal state (Completed OR Failed) so a fault is captured fast
# instead of burning the full timeout waiting for a "Completed" that won't come.
echo "[lab-13a] waiting for gang to reach a terminal state..."
gang_state="(timeout)"
for i in $(seq 1 120); do   # up to ~600s
  c=$(kubectl get jobset/nccl-gang -o jsonpath='{range .status.conditions[*]}{.type}={.status} {end}' 2>/dev/null || true)
  case "$c" in
    *Completed=True*) gang_state="Completed"; break ;;
    *Failed=True*)    gang_state="Failed";    break ;;
  esac
  sleep 5
done
echo "gang terminal state: ${gang_state} (conditions: ${c})" | tee "${OUT}/gang_complete.txt"
# Capture per-pod logs LIVE (pods are GC'd once the JobSet is deleted).
cap_run lab-13 gang_allreduce_result.txt -- bash -c "echo '### per-worker log (expect rank-0 line: GANG all_reduce OK value=24.0)'; for p in \$(kubectl --context ${KUBE_CONTEXT} get pods -l jobset.sigs.k8s.io/jobset-name=nccl-gang -o name 2>/dev/null); do echo \"-- \$p --\"; kubectl --context ${KUBE_CONTEXT} logs \$p 2>&1 | grep -E 'GANG|node_rank|host=|FATAL|Error|error' | tail -6 || echo '(no log)'; done"

# --- 4. Over-quota gang: Kueue gates it (no pods) ----------------------------
echo "[lab-13a] submitting 32-GPU over-quota JobSet (expect gated, no pods)"
kubectl apply -f "$OVERQUOTA"
sleep 10
cap_run lab-13 gang_overquota_gate.txt -- bash -c "echo '### over-quota (32 GPU) workload: QuotaReserved=False, no pods'; kubectl --context ${KUBE_CONTEXT} get workloads -n default -o json | python3 -c 'import json,sys
for w in json.load(sys.stdin)[\"items\"]:
    if \"overquota\" in w[\"metadata\"][\"name\"]:
        conds=[(c[\"type\"],c[\"status\"],c.get(\"reason\",\"\")) for c in w.get(\"status\",{}).get(\"conditions\",[])]
        print(\"  workload:\", w[\"metadata\"][\"name\"], \"conds=\", conds)'; echo; echo '### pods for over-quota jobset (expect NONE):'; kubectl --context ${KUBE_CONTEXT} get pods -l jobset.sigs.k8s.io/jobset-name=nccl-gang-overquota 2>&1 | tail -2"
kubectl delete -f "$OVERQUOTA" --wait=false --ignore-not-found

echo "[lab-13a] ===== gang result ====="
cat "${OUT}/gang_allreduce_result.txt" 2>/dev/null || true

cap_verify_provenance "lab-13" "assets/lab-13" "${NODES[0]},${NODES[1]},${NODES[2]}" \
  "asia-east1-c 24-GPU/3-node gang via JobSet+Kueue: admitted as one Workload, placed 1 pod/node, all-reduce value=24.0; 32-GPU over-quota gated"
echo "[lab-13a] done. (JobSet/queues deleted + holder restored by EXIT trap)"
