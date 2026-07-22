#!/usr/bin/env bash
# Manual c10d launcher for a 2-node all-reduce benchmark — one process per GPU.
#
# Bypasses torchrun's elastic rendezvous (fragile across two independently
# started pods) by setting RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT
# directly and letting torch.distributed's default env:// init build the
# TCPStore (rank 0 hosts it). Usage:
#   bash launch_node.sh <NODE_RANK> <MASTER_ADDR> [MASTER_PORT] [NPROC]
# The Python entrypoint defaults to allreduce_bench.py; override + pass extra
# args via the BENCH_SCRIPT / BENCH_ARGS env vars (used by the DDP/FSDP lab).
set -uo pipefail
NODE_RANK="${1:?node_rank}"; MASTER_ADDR="${2:?master_addr}"
MASTER_PORT="${3:-29520}"; NPROC="${4:-8}"
NNODES="${NNODES:-2}"
BENCH_SCRIPT="${BENCH_SCRIPT:-/workspace/allreduce_bench.py}"
BENCH_ARGS="${BENCH_ARGS:-}"

export WORLD_SIZE=$((NNODES * NPROC))
export MASTER_ADDR MASTER_PORT
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export BENCH_MAX_EXP="${BENCH_MAX_EXP:-30}"
export BENCH_ITERS="${BENCH_ITERS:-15}"

echo "[launch] node_rank=$NODE_RANK world=$WORLD_SIZE master=$MASTER_ADDR:$MASTER_PORT nproc=$NPROC"
pids=()
for i in $(seq 0 $((NPROC - 1))); do
  RANK=$((NODE_RANK * NPROC + i)) LOCAL_RANK=$i \
    python "$BENCH_SCRIPT" $BENCH_ARGS > "/workspace/w_$((NODE_RANK * NPROC + i)).log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[launch] node_rank=$NODE_RANK done rc=$rc"
exit $rc
