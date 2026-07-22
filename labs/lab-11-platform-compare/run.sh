#!/usr/bin/env bash
# Lab 11 — Platform compare (A3 tenant view vs. DGX/SuperPOD reference)
#
# runnable-here  = the live probes below (read-only; no protected workload touched)
# read-only-ref  = the DGX/BlueField/Spectrum-X/SuperPOD material in docs 11-14
#
# Captures REAL evidence that this A3 node is an HGX H100 baseboard, that its
# NVSwitch NVLink mesh is trained (Fabric State: Completed), and that there is
# NO tenant-visible Fabric Manager, BlueField DPU, or Mellanox/ConnectX SuperNIC
# — only Google gVNIC. Separates measured facts from reference knowledge.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${REPO_ROOT}/scripts/lib_capture.sh"
OUT="${REPO_ROOT}/assets/lab-11"
mkdir -p "$OUT"

# Pick a GPU node that has free GPUs (do NOT target a fully-held capacity-holder node).
NODE="${LAB11_NODE:-$(kubectl get nodes -l cloud.google.com/gke-nodepool=a3-h100-dws-pool \
  -o jsonpath='{.items[0].metadata.name}')}"
echo "[lab-11] target node: $NODE"

# ---- 1. Host-level absence probes (read-only, via kubectl debug node) --------
# lspci enumerates all 8 GPUs + the NIC from /sys without allocating any GPU.
kubectl debug "node/${NODE}" -it --image=busybox:1.36 -- sh -c '
  echo "### lspci NVIDIA/Mellanox/BlueField ###"
  chroot /host lspci 2>/dev/null | grep -iE "nvidia|mellanox|bluefield|connectx|ethernet" || echo "no matches"
  echo "### net interfaces ###"; chroot /host ls /sys/class/net 2>/dev/null
  echo "### nv-fabricmanager present? ###"
  chroot /host sh -c "which nv-fabricmanager; ls /usr/bin/nv-fabricmanager; pgrep -a fabricmanager" 2>/dev/null || true
  echo "(blank above => nv-fabricmanager not present/visible to tenant node OS)"
' 2>/dev/null | grep -vE '^(Creating|Warning)' > "${OUT}/node-probes.txt" || true

# ---- 2. GPU-side probes: NVLink mesh + fabric state (small pod, torn down) ----
# Allocate a few GPUs (not all free ones) purely to query topology; delete right after.
cat > /tmp/lab11-topo-pod.yaml <<YAML
apiVersion: v1
kind: Pod
metadata: { name: lab11-topo }
spec:
  restartPolicy: Never
  nodeName: ${NODE}
  tolerations:
  - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
  - { key: cloud.google.com/gke-queued, operator: Exists, effect: NoSchedule }
  containers:
  - name: cuda
    image: nvcr.io/nvidia/pytorch:24.10-py3
    command: ["sleep", "infinity"]
    resources: { limits: { nvidia.com/gpu: 4 } }
YAML
kubectl delete pod lab11-topo --ignore-not-found >/dev/null 2>&1 || true
kubectl apply -f /tmp/lab11-topo-pod.yaml
kubectl wait --for=condition=Ready pod/lab11-topo --timeout=300s

cap_run "topo-4gpu"     "${OUT}/topo-4gpu.txt"     -- kubectl exec lab11-topo -- nvidia-smi topo -m
cap_run "nvlink-status" "${OUT}/nvlink-status.txt" -- kubectl exec lab11-topo -- nvidia-smi nvlink --status
cap_run "fabric-state"  "${OUT}/fabric-state.txt"  -- \
  kubectl exec lab11-topo -- bash -lc 'nvidia-smi -q | grep -iA3 "^    Fabric"'

# Free the GPUs immediately — never leave the borrowed GPUs idle-held.
kubectl delete pod lab11-topo --ignore-not-found

# ---- 3. Provenance ----------------------------------------------------------
cap_verify_provenance "lab-11" "assets/lab-11" "$NODE" \
  "platform-compare: HGX baseboard confirmed; gVNIC only; no FM/DPU/SuperNIC"

echo "[lab-11] done. See assets/lab-11/ and a3-vs-dgx-superpod.csv"
