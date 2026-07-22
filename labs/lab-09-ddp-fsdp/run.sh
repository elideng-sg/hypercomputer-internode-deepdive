#!/usr/bin/env bash
# Lab 09 — 2-node DDP and FSDP distributed training over the inter-node path.
#
# Runs the same synthetic-model training loop under two parallelism strategies
# across TWO A3 nodes (8 GPUs each = 16 ranks):
#   * DDP   -> gradient all-reduce every step (bucketed)
#   * FSDP  -> params + grads sharded; all-gather (fwd) + reduce-scatter (bwd)
# and, for DDP, captures a torch.profiler trace so the communication op that
# dominates the step can be read directly.
#
# Reuses lab-06's manual c10d launcher (launch_node.sh) — no MPI, no SSH — via
# its BENCH_SCRIPT/BENCH_ARGS hooks. Same rendezvous that made the 2-node NCCL
# sweep robust across two independently-launched pods.
#
# GPU SAFETY: needs 8 GPUs on EACH of two nodes. Both nodes are normally
# occupied (capacity holder + vLLM); this ran in a planned window with two
# 8-GPU workbench pods, holder + vLLM restored immediately afterward.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
OUT="${REPO_ROOT}/assets/lab-09"; mkdir -p "$OUT"
LAB09="${REPO_ROOT}/labs/lab-09-ddp-fsdp"
LAB06="${REPO_ROOT}/labs/lab-06-2node-nccl-collectives"

POD_A="${LAB09_POD_A:-nccl-workbench-a}"   # node_rank 0, hosts the c10d store
POD_B="${LAB09_POD_B:-nccl-workbench-b}"   # node_rank 1
PORT="${LAB09_PORT:-29531}"

MASTER_IP="$(kubectl exec "$POD_A" -- bash -lc 'hostname -I | awk "{print \$1}"')"
NODE_A="$(kubectl get pod "$POD_A" -o jsonpath='{.spec.nodeName}')"
NODE_B="$(kubectl get pod "$POD_B" -o jsonpath='{.spec.nodeName}')"
echo "[lab-09] rank0=$POD_A ($NODE_A) rank1=$POD_B ($NODE_B) master=$MASTER_IP:$PORT"

# Stage the training script + the (shared) launcher into both pods.
for P in "$POD_A" "$POD_B"; do
  kubectl cp "${LAB09}/train_ddp_fsdp.py" "${P}:/workspace/train_ddp_fsdp.py"
  kubectl cp "${LAB06}/launch_node.sh"    "${P}:/workspace/launch_node.sh"
done

# run_mode <mode> <extra-train-args> <output-file>
run_mode() {
  local mode="$1" extra="$2" outfile="$3"
  echo "[lab-09] === $mode ==="
  local ENVV="NCCL_SOCKET_IFNAME=eth0 BENCH_SCRIPT=/workspace/train_ddp_fsdp.py \
BENCH_ARGS='--mode ${mode} ${extra}'"
  kubectl exec "$POD_A" -- bash -lc "cd /workspace && ${ENVV} bash launch_node.sh 0 ${MASTER_IP} ${PORT} 8" &
  sleep 6
  kubectl exec "$POD_B" -- bash -lc "cd /workspace && ${ENVV} bash launch_node.sh 1 ${MASTER_IP} ${PORT} 8"
  wait
  kubectl exec "$POD_A" -- bash -lc 'cat /workspace/w_0.log' > "$outfile"
  kubectl exec "$POD_A" -- bash -lc 'rm -f /workspace/w_*.log'
}

# 1. DDP (with profiler → trace + top-ops table) and FSDP.
run_mode ddp  "--steps 30 --profile --trace-out /workspace/trace_rank0.json" "${OUT}/ddp_2node.txt"
kubectl exec "$POD_A" -- bash -lc 'gzip -c /workspace/trace_rank0.json' > "${OUT}/ddp_trace_rank0.json.gz" || true
# The profiler key_averages() table is printed into ddp_2node.txt; split it out.
grep -E 'Name|allreduce|nccl|AdamW|aten::mm|ncclDevKernel|CUDA time total|ProfilerStep|-----' \
  "${OUT}/ddp_2node.txt" > "${OUT}/ddp_profiler_top_ops.txt" || true

run_mode fsdp "--steps 30" "${OUT}/fsdp_2node.txt"

cap_verify_provenance "lab-09" "assets/lab-09" "${NODE_A},${NODE_B}" \
  "2-node 16-GPU DDP+FSDP training; DDP profiler trace captured (all-reduce dominates step)"
echo "[lab-09] done. See assets/lab-09/"
