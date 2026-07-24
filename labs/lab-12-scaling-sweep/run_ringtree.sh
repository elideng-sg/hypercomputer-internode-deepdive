#!/usr/bin/env bash
# Lab 12b — Ring vs. Tree all-reduce at 24 GPUs (3 nodes).
#
# Forces NCCL_ALGO=Ring then NCCL_ALGO=Tree at 24 GPUs and sweeps message size
# to locate the small-message crossover. WHY 3 NODES: at 2 nodes tree depth is
# trivial and NCCL's own note ("algorithm matters far less than the link") holds;
# at 3 nodes tree depth grows to 2 and the ring/tree divergence becomes real and
# measurable — a phenomenon 2 nodes cannot show. Feeds doc-15's ring/tree section.
#
# GPU SAFETY: identical guarded borrow window to run.sh — scale gpu-holder 3→0,
# occupy with 3 workbench pods, run, then delete pods + restore holder via EXIT
# trap. No node drained/deleted (Flex-safe).
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export KUBE_CONTEXT="${KUBE_CONTEXT:-gke_hdlab-elideng_asia-east1-c_hypercomputer-a3-asiaeast1}"
export LAB_NODEPOOL="${LAB_NODEPOOL:-a3-high-flex-pool}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
OUT="${REPO_ROOT}/assets/lab-12"; mkdir -p "$OUT"
LAB06="${REPO_ROOT}/labs/lab-06-2node-nccl-collectives"

PORT="${LAB12_PORT:-29522}"
IMAGE="${LAB12_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
PODS=(nccl-wb-a nccl-wb-b nccl-wb-c)

mapfile -t NODES < <(cap_nodes | sort)
[ "${#NODES[@]}" -ge 3 ] || { echo "[lab-12b] FATAL: need 3 nodes, found ${#NODES[@]}" >&2; exit 1; }
echo "[lab-12b] nodes: ${NODES[*]}"

cleanup() {
  echo "[lab-12b] cleanup: freeing workbenches and restoring gpu-holder to 3"
  kubectl delete pod "${PODS[@]}" --wait=false --ignore-not-found 2>/dev/null || true
  kubectl scale deploy gpu-holder --replicas=3 2>/dev/null || true
}
trap cleanup EXIT

echo "[lab-12b] scaling gpu-holder 3->0"
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
echo "[lab-12b] waiting for workbench pods..."
kubectl wait --for=condition=Ready pod "${PODS[@]}" --timeout=300s
for P in "${PODS[@]}"; do
  kubectl cp "${LAB06}/allreduce_bench.py" "${P}:/workspace/allreduce_bench.py"
  kubectl cp "${LAB06}/launch_node.sh"     "${P}:/workspace/launch_node.sh"
done

# Run a 24-GPU sweep with a forced NCCL_ALGO. $1 = Ring|Tree
run_algo() {
  local algo="$1" master_ip pids=() node_r pod
  master_ip="$(kubectl exec "${PODS[0]}" -- bash -lc 'hostname -I | awk "{print \$1}"')"
  local envv="NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH NCCL_ALGO=${algo} NCCL_PROTO=Simple BENCH_MAX_EXP=30 BENCH_ITERS=15"
  echo "[lab-12b] === NCCL_ALGO=${algo}, 24 GPU, master=${master_ip}:${PORT} ==="
  for node_r in 0 1 2; do
    pod="${PODS[$node_r]}"
    kubectl exec "$pod" -- bash -lc \
      "cd /workspace && NNODES=3 ${envv} bash launch_node.sh ${node_r} ${master_ip} ${PORT} 8" &
    pids+=($!)
    [ "$node_r" -eq 0 ] && sleep 6
  done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  local lc; lc="$(echo "$algo" | tr '[:upper:]' '[:lower:]')"
  kubectl exec "${PODS[0]}" -- bash -lc 'cat /workspace/w_0.log' > "${OUT}/ringtree_${lc}_full.log"
  grep -E '^#|^\s+[0-9]' "${OUT}/ringtree_${lc}_full.log" > "${OUT}/ringtree_${lc}.txt" || true
  # confirm NCCL honored the forced algo
  grep -iE "Connected all|${algo}|Algorithm|NCCL_ALGO" "${OUT}/ringtree_${lc}_full.log" | head -5 || true
  kubectl exec "${PODS[0]}" -- bash -lc 'rm -f /workspace/w_*.log' || true
  echo "[lab-12b] ${algo} rc=${rc}"
}

run_algo Ring
run_algo Tree

# Build a side-by-side busbw table (ring vs tree) for the crossover.
{
  echo "size_bytes,ring_busbw_GBps,tree_busbw_GBps"
  paste -d' ' \
    <(awk '/^ / {print $1"|"$5}' "${OUT}/ringtree_ring.txt") \
    <(awk '/^ / {print $5}'      "${OUT}/ringtree_tree.txt") \
  | awk '{split($1,a,"|"); print a[1]","a[2]","$2}'
} > "${OUT}/ringtree_crossover.csv" 2>/dev/null || true
echo "[lab-12b] ringtree_crossover.csv:"; cat "${OUT}/ringtree_crossover.csv" 2>/dev/null | head -40

cap_verify_provenance "lab-12" "assets/lab-12" "${NODES[0]},${NODES[1]},${NODES[2]}" \
  "asia-east1-c 3-node ring-vs-tree: NCCL_ALGO Ring/Tree 24-GPU all-reduce sweep"
echo "[lab-12b] done. (holder restored by EXIT trap)"
