#!/usr/bin/env bash
# Lab 06 — 2-node, 16-GPU NCCL all-reduce over the real inter-node path.
#
# Runs a PyTorch/NCCL all-reduce bandwidth sweep across TWO A3 nodes (8 GPUs
# each = 16 ranks) and captures the NCCL transport line, so the actual data
# path is recorded rather than assumed. Reports busbw vs. message size using
# the same busbw = algbw * 2(n-1)/n definition as nccl-tests.
#
# Why torch.distributed and not `mpirun all_reduce_perf`? The NGC image ships
# nccl-tests but no sshd, so a 2-node MPI launch would need SSH plumbing
# between pods. torch.distributed's c10d env:// rendezvous needs neither MPI
# nor SSH and uses the identical NCCL library — same measurement, robust across
# two independently-launched pods. (Single-node lab-04 uses nccl-tests directly.)
#
# GPU SAFETY: needs 8 GPUs on EACH of two nodes. In this environment both nodes
# are normally occupied (a capacity holder + the vLLM service); this ran during
# a planned window with two 8-GPU "workbench" pods, both freed and the holder +
# vLLM restored immediately afterward.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
OUT="${REPO_ROOT}/assets/lab-06"; mkdir -p "$OUT"
LAB="${REPO_ROOT}/labs/lab-06-2node-nccl-collectives"

POD_A="${LAB06_POD_A:-nccl-workbench-a}"   # node_rank 0, hosts the c10d store
POD_B="${LAB06_POD_B:-nccl-workbench-b}"   # node_rank 1
PORT="${LAB06_PORT:-29520}"

MASTER_IP="$(kubectl exec "$POD_A" -- bash -lc 'hostname -I | awk "{print \$1}"')"
NODE_A="$(kubectl get pod "$POD_A" -o jsonpath='{.spec.nodeName}')"
NODE_B="$(kubectl get pod "$POD_B" -o jsonpath='{.spec.nodeName}')"
echo "[lab-06] rank0=$POD_A ($NODE_A) rank1=$POD_B ($NODE_B) master=$MASTER_IP:$PORT"

# Stage the benchmark + launcher into both pods.
for P in "$POD_A" "$POD_B"; do
  kubectl cp "${LAB}/allreduce_bench.py" "${P}:/workspace/allreduce_bench.py"
  kubectl cp "${LAB}/launch_node.sh"     "${P}:/workspace/launch_node.sh"
done

# Launch rank0 (master) first, then rank1. NCCL_DEBUG=INFO records the transport.
ENVV='NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET BENCH_MAX_EXP=30 BENCH_ITERS=15'
kubectl exec "$POD_A" -- bash -lc "cd /workspace && ${ENVV} bash launch_node.sh 0 ${MASTER_IP} ${PORT} 8" &
sleep 6
kubectl exec "$POD_B" -- bash -lc "cd /workspace && ${ENVV} bash launch_node.sh 1 ${MASTER_IP} ${PORT} 8"
wait

# Pull rank-0 results + transport evidence.
kubectl exec "$POD_A" -- bash -lc 'cat /workspace/w_0.log' > "${OUT}/allreduce_2node_rank0_full.log"
grep -E '^#|^\s+[0-9]' "${OUT}/allreduce_2node_rank0_full.log" > "${OUT}/allreduce_2node.txt"
grep -iE 'NET/IB|NET/Socket|Using network|GPU Direct|NCCL version|Channel|Trees|Rings' \
  "${OUT}/allreduce_2node_rank0_full.log" | head -40 > "${OUT}/nccl_transport.txt"
kubectl exec "$POD_A" -- bash -lc 'rm -f /workspace/w_*.log'

cap_verify_provenance "lab-06" "assets/lab-06" "${NODE_A},${NODE_B}" \
  "2-node 16-GPU all-reduce; transport recorded in nccl_transport.txt"
echo "[lab-06] done. See assets/lab-06/"
