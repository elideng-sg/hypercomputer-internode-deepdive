#!/usr/bin/env bash
# Lab 12c — DDP & FSDP training scaling at 8 / 16 / 24 GPUs (1 / 2 / 3 nodes).
#
# Reuses lab-09's train_ddp_fsdp.py (1B-param synthetic model) via launch_node.sh's
# BENCH_SCRIPT/BENCH_ARGS hooks, at three node counts on asia-east1-c. Captures:
#   * WEAK scaling  — per-rank batch fixed; global batch grows with GPU count.
#                     Ideal: samples/s scales linearly. Efficiency = how much the
#                     inter-node all-reduce (which we measured degrading in 12a)
#                     erodes that as the gang grows.
#   * STRONG scaling — global batch FIXED (384); per-rank batch shrinks. Ideal:
#                     step time falls 1/N. Efficiency = realized speedup / ideal.
#
# WHY 3 NODES: scaling efficiency is a *slope*; a single 2-node step-time is one
# point and cannot express it. Feeds doc-15's scaling-efficiency section.
#
# GPU SAFETY: identical guarded borrow window to run.sh — scale gpu-holder 3->0,
# occupy with 3 workbench pods, EXIT-trap restores the hold. No node drained
# (Flex-safe).
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
OUT="${REPO_ROOT}/assets/lab-12"; mkdir -p "$OUT"
LAB06="${REPO_ROOT}/labs/lab-06-2node-nccl-collectives"
LAB09="${REPO_ROOT}/labs/lab-09-ddp-fsdp"

PORT="${LAB12_PORT:-29523}"
IMAGE="${LAB12_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
STEPS="${LAB12C_STEPS:-30}"
GLOBAL_BATCH="${LAB12C_GLOBAL_BATCH:-384}"   # divisible by 8/16/24
PER_RANK_BATCH="${LAB12C_PER_RANK_BATCH:-16}"
PODS=(nccl-wb-a nccl-wb-b nccl-wb-c)

mapfile -t NODES < <(cap_nodes | sort)
[ "${#NODES[@]}" -ge 3 ] || { echo "[lab-12c] FATAL: need 3 nodes, found ${#NODES[@]}" >&2; exit 1; }
echo "[lab-12c] nodes: ${NODES[*]}"

cleanup() {
  echo "[lab-12c] cleanup: freeing workbenches and restoring gpu-holder to 3"
  kubectl delete pod "${PODS[@]}" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

echo "[lab-12c] scaling gpu-holder 3->0"
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
echo "[lab-12c] waiting for workbench pods..."
kubectl wait --for=condition=Ready pod "${PODS[@]}" --timeout=300s
for P in "${PODS[@]}"; do
  kubectl cp "${LAB09}/train_ddp_fsdp.py" "${P}:/workspace/train_ddp_fsdp.py"
  kubectl cp "${LAB06}/launch_node.sh"    "${P}:/workspace/launch_node.sh"
done

# run_train <nnodes> <mode> <per-rank-batch> <outfile>
run_train() {
  local nnodes="$1" mode="$2" batch="$3" outfile="$4"
  local world=$((nnodes * 8)) master_ip pids=() node_r pod
  master_ip="$(kubectl exec "${PODS[0]}" -- bash -lc 'hostname -I | awk "{print \$1}"')"
  echo "[lab-12c] === ${mode} ${world}-GPU batch/rank=${batch} ==="
  local envv="NCCL_SOCKET_IFNAME=eth0 BENCH_SCRIPT=/workspace/train_ddp_fsdp.py BENCH_ARGS='--mode ${mode} --steps ${STEPS} --batch ${batch}'"
  for node_r in $(seq 0 $((nnodes - 1))); do
    pod="${PODS[$node_r]}"
    kubectl exec "$pod" -- bash -lc \
      "cd /workspace && NNODES=${nnodes} ${envv} bash launch_node.sh ${node_r} ${master_ip} ${PORT} 8" &
    pids+=($!)
    [ "$node_r" -eq 0 ] && sleep 6
  done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  kubectl exec "${PODS[0]}" -- bash -lc 'cat /workspace/w_0.log' > "$outfile"
  kubectl exec "${PODS[0]}" -- bash -lc 'rm -f /workspace/w_*.log' || true
  echo "[lab-12c] ${mode} ${world}-GPU rc=${rc}"
}

# --- WEAK scaling: per-rank batch fixed, global batch grows ------------------
for nn in 1 2 3; do
  w=$((nn * 8))
  run_train "$nn" ddp  "$PER_RANK_BATCH" "${OUT}/train_weak_ddp_${w}gpu.txt"
  run_train "$nn" fsdp "$PER_RANK_BATCH" "${OUT}/train_weak_fsdp_${w}gpu.txt"
done

# --- STRONG scaling (DDP): global batch fixed, per-rank shrinks --------------
for nn in 1 2 3; do
  w=$((nn * 8)); b=$((GLOBAL_BATCH / w))
  run_train "$nn" ddp "$b" "${OUT}/train_strong_ddp_${w}gpu.txt"
done

# --- Parse steady-state lines into efficiency CSVs ---------------------------
# steady-state line: "# steady-state: <ms> ms/step  <samples> samples/s (global)"
get_ms()  { grep -m1 'steady-state' "$1" 2>/dev/null | sed -E 's/.*: ([0-9.]+) ms.*/\1/'; }
get_sps() { grep -m1 'steady-state' "$1" 2>/dev/null | sed -E 's|.*ms/step[[:space:]]+([0-9.]+) samples.*|\1|'; }

{
  echo "mode,gpus,nodes,step_ms,samples_per_s,weak_scaling_eff_pct"
  for mode in ddp fsdp; do
    base_sps=""
    for nn in 1 2 3; do
      w=$((nn*8)); f="${OUT}/train_weak_${mode}_${w}gpu.txt"
      [ -s "$f" ] || continue
      ms=$(get_ms "$f"); sps=$(get_sps "$f")
      [ -z "$base_sps" ] && base_sps="$sps"
      eff=$(awk -v s="$sps" -v b="$base_sps" -v n="$nn" 'BEGIN{if(b>0)printf "%.1f",(s/(b*n))*100; else print "NA"}')
      echo "${mode},${w},${nn},${ms},${sps},${eff}"
    done
  done
} > "${OUT}/train_weak_scaling.csv"

{
  echo "mode,gpus,nodes,step_ms,strong_scaling_eff_pct"
  base_ms=""
  for nn in 1 2 3; do
    w=$((nn*8)); f="${OUT}/train_strong_ddp_${w}gpu.txt"
    [ -s "$f" ] || continue
    ms=$(get_ms "$f")
    [ -z "$base_ms" ] && base_ms="$ms"
    # strong: ideal step_ms = base_ms / n; efficiency = ideal/actual
    eff=$(awk -v m="$ms" -v b="$base_ms" -v n="$nn" 'BEGIN{if(m>0)printf "%.1f",((b/n)/m)*100; else print "NA"}')
    echo "ddp,${w},${nn},${ms},${eff}"
  done
} > "${OUT}/train_strong_scaling.csv"

echo "[lab-12c] weak scaling:";   cat "${OUT}/train_weak_scaling.csv"
echo "[lab-12c] strong scaling:"; cat "${OUT}/train_strong_scaling.csv"

cap_verify_provenance "lab-12" "assets/lab-12" "${NODES[0]},${NODES[1]},${NODES[2]}" \
  "asia-east1-c 3-node training scaling: DDP+FSDP weak & strong at 8/16/24 GPUs"
echo "[lab-12c] done. (holder restored by EXIT trap)"
