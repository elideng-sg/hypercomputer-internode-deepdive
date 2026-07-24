#!/usr/bin/env bash
# Lab 12a — all-reduce scaling sweep at 8 / 16 / 24 GPUs (1 / 2 / 3 nodes).
#
# Captures busbw vs. message size at THREE node counts on ONE cluster
# (asia-east1-c, the 3-node a3-high-flex-pool) so the two-point "cliff" becomes
# a real curve. Reuses lab-06's launch_node.sh + allreduce_bench.py verbatim
# (same NCCL library, same busbw = algbw*2(n-1)/n definition) so the 8/16/24-GPU
# points are directly comparable to lab-04 (intra-node) and lab-06 (2-node).
#
# WHY 3 NODES: one inter-node point proves a cliff; the *shape* (does peak busbw
# hold across a 3rd TCP hop? does the latency floor grow?) needs a 3rd point. At
# 24 GPUs the NCCL ring crosses THREE node boundaries — captured in the transport
# dump. See docs/15-scaling-shape-of-the-cliff.md.
#
# ─────────────────────────────────────────────────────────────────────────────
# GPU SAFETY (read before running):
#   This cluster's 24 H100s are normally fully held by the `gpu-holder`
#   Deployment (3×8). This lab performs a GUARDED, gap-free hold handoff:
#     1. scale gpu-holder 3→0   (frees the 24 GPUs)
#     2. create 3 workbench pods (they immediately occupy the freed GPUs —
#        the nodes are never idle; the workbench IS the occupancy for the window)
#     3. run the sweep
#     4. delete workbenches + scale gpu-holder back to 3   (via EXIT trap, so the
#        hold is ALWAYS restored even if the sweep fails midway)
#   No node is drained or deleted (Flex-safe). Mirrors lab-06's planned-window
#   pattern. The trap is the safety net: on any exit the holder is re-armed.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Target the 3-node asia-east1-c cluster (labs 01-11 default to us-central1).
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
OUT="${REPO_ROOT}/assets/lab-12"; mkdir -p "$OUT"
LAB06="${REPO_ROOT}/labs/lab-06-2node-nccl-collectives"

PORT="${LAB12_PORT:-29521}"
IMAGE="${LAB12_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
# Bound wall-clock on the TCP fabric: 8 B .. 1 GiB (matches lab-06's sweep).
ENVV='NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH BENCH_MAX_EXP=30 BENCH_ITERS=15'
PODS=(nccl-wb-a nccl-wb-b nccl-wb-c)

# --- Enumerate the 3 flex nodes (stable order) ------------------------------
mapfile -t NODES < <(cap_nodes | sort)
if [ "${#NODES[@]}" -lt 3 ]; then
  echo "[lab-12] FATAL: need 3 GPU nodes in $LAB_NODEPOOL, found ${#NODES[@]}" >&2
  exit 1
fi
echo "[lab-12] nodes: ${NODES[0]} ${NODES[1]} ${NODES[2]}"

# --- Safety net: ALWAYS free workbenches + restore the holder on exit --------
cleanup() {
  echo "[lab-12] cleanup: freeing workbenches and restoring gpu-holder to 3"
  kubectl delete pod "${PODS[@]}" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

# --- Hold handoff: free the GPUs, then occupy them with workbenches ----------
echo "[lab-12] scaling gpu-holder 3->0 to free 24 GPUs for the borrow window"
kubectl scale deploy gpu-holder --replicas=0
kubectl wait --for=delete pod -l app=gpu-holder --timeout=180s || true

gen_workbench() { # gen_workbench <pod> <node>
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
echo "[lab-12] waiting for 3 workbench pods to be Ready..."
kubectl wait --for=condition=Ready pod "${PODS[@]}" --timeout=300s

# --- Stage the (unmodified) lab-06 benchmark + launcher into each pod --------
for P in "${PODS[@]}"; do
  kubectl cp "${LAB06}/allreduce_bench.py" "${P}:/workspace/allreduce_bench.py"
  kubectl cp "${LAB06}/launch_node.sh"     "${P}:/workspace/launch_node.sh"
done

# --- One sweep at a given node count -----------------------------------------
# run_config <nnodes> -> writes assets/lab-12/allreduce_<world>gpu(.txt|_full.log)
run_config() {
  local nnodes="$1"; local world=$((nnodes * 8))
  local master_ip node_r pod pids=()
  master_ip="$(kubectl exec "${PODS[0]}" -- bash -lc 'hostname -I | awk "{print \$1}"')"
  echo "[lab-12] === ${world}-GPU sweep (${nnodes} node(s)), master=${master_ip}:${PORT} ==="
  for node_r in $(seq 0 $((nnodes - 1))); do
    pod="${PODS[$node_r]}"
    kubectl exec "$pod" -- bash -lc \
      "cd /workspace && NNODES=${nnodes} ${ENVV} bash launch_node.sh ${node_r} ${master_ip} ${PORT} 8" &
    pids+=($!)
    [ "$node_r" -eq 0 ] && sleep 6   # let rank0 stand up the c10d store first
  done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done

  # rank-0 log holds the busbw table + (at INFO) the transport/ring lines.
  kubectl exec "${PODS[0]}" -- bash -lc 'cat /workspace/w_0.log' \
    > "${OUT}/allreduce_${world}gpu_full.log"
  grep -E '^#|^\s+[0-9]' "${OUT}/allreduce_${world}gpu_full.log" \
    > "${OUT}/allreduce_${world}gpu.txt" || true
  kubectl exec "${PODS[0]}" -- bash -lc 'rm -f /workspace/w_*.log' || true
  echo "[lab-12] ${world}-GPU sweep rc=${rc}"
}

run_config 1   # 8 GPU  — single node (intra-node NVLink baseline on this cluster)
run_config 2   # 16 GPU — one inter-node TCP hop
run_config 3   # 24 GPU — ring crosses THREE node boundaries

# --- Capture the 24-GPU transport + ring evidence ----------------------------
grep -iE 'NET/IB|NET/Socket|NET/GPUDirect|Using network|GPU Direct|NCCL version|Channel|Trees|Rings|via NET' \
  "${OUT}/allreduce_24gpu_full.log" | head -60 > "${OUT}/nccl_transport_24gpu.txt" || true

# --- Build the scaling CSV (peak busbw + latency floor per node count) --------
# peak busbw = max busbw column; latency floor = time_ms of the smallest message.
{
  echo "gpus,nodes,peak_busbw_GBps,latency_floor_ms"
  for nn in 1 2 3; do
    w=$((nn * 8)); f="${OUT}/allreduce_${w}gpu.txt"
    [ -s "$f" ] || continue
    peak=$(awk '/^ / {if ($5+0>m) m=$5} END{printf "%.2f", m}' "$f")
    floor=$(awk '/^ / {print $3; exit}' "$f")
    echo "${w},${nn},${peak},${floor}"
  done
} > "${OUT}/scaling_curve.csv"
echo "[lab-12] scaling_curve.csv:"; cat "${OUT}/scaling_curve.csv"

# --- Plot busbw vs message size for all three node counts (one figure) -------
python3 "${REPO_ROOT}/scripts/lib_plot.py" \
  --csv "${OUT}/scaling_curve.csv" --x gpus --y peak_busbw_GBps \
  --out "${OUT}/scaling_peak_busbw.png" --kind line \
  --title "All-reduce peak busbw vs GPU count (asia-east1-c, TCP)" \
  --xlabel "GPUs" --ylabel "peak busbw (GB/s)" 2>/dev/null || \
  echo "[lab-12] (plot skipped — lib_plot.py unavailable)"

cap_verify_provenance "lab-12" "assets/lab-12" "${NODES[0]},${NODES[1]},${NODES[2]}" \
  "asia-east1-c 3-node scaling sweep: 8/16/24-GPU all-reduce; transport in nccl_transport_24gpu.txt"
echo "[lab-12] done. See assets/lab-12/  (holder restored by EXIT trap)"
