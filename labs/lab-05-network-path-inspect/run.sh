#!/usr/bin/env bash
# lab-05: characterize the ACTUAL inter-node network path on this A3 cluster.
# 0-GPU lab — never requests nvidia.com/gpu, so it is safe to run alongside
# DWS holders and any GPU workload. Pins iperf3 to the two GPU nodes by hostname.
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib_capture.sh"
LAB=lab-05
mkdir -p "$ASSETS/$LAB"

# --- Enumerate the two GPU nodes -------------------------------------------
mapfile -t NODES < <(cap_nodes)
NODE0="${NODES[0]}"; NODE1="${NODES[1]}"
echo "GPU nodes: $NODE0 , $NODE1"

# --- 1. Node NIC / GPUDirect networking annotations -------------------------
cap_run $LAB node-annotations.txt -- bash -c "for n in $NODE0 $NODE1; do echo \"### \$n\"; kubectl get node \$n -o jsonpath='{.metadata.annotations}' | tr ',' '\n' | grep -iE 'networking.gke.io|gpu|nic|tcpx|rdma' || echo '(no matching annotations)'; echo; done"

# --- 2. Extended resources (is a GPU-NIC resource present?) -----------------
cap_run $LAB allocatable.txt -- bash -c "for n in $NODE0 $NODE1; do echo \"### \$n\"; kubectl describe node \$n | sed -n '/Allocatable/,/System Info/p'; echo; done"

# --- 3. Installed network / NCCL-plugin DaemonSets --------------------------
cap_run $LAB net-daemonsets.txt -- bash -c "kubectl get ds -A -o wide | grep -iE 'tcpx|tcpxo|fastsocket|rdma|nccl|gpudirect|network' || echo '(no tcpx/tcpxo/fastsocket/rdma/nccl network DaemonSets found)'"

# --- 4. Per-node link inventory via a privileged host-network probe pod -----
# hostNetwork + 0 GPU: reads the node's NICs without consuming any resource.
for i in 0 1; do
  NODE="${NODES[$i]}"
  POD="net-probe-$i"
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: $POD
spec:
  hostNetwork: true
  nodeName: $NODE
  tolerations:
  - operator: Exists
  restartPolicy: Never
  containers:
  - name: probe
    image: networkstatic/iperf3:latest
    command: ["sleep", "600"]
EOF
done
kubectl wait --for=condition=Ready pod/net-probe-0 pod/net-probe-1 --timeout=180s
cap_run $LAB links.txt -- bash -c "for p in net-probe-0 net-probe-1; do echo \"### \$p\"; kubectl exec \$p -- ip -br link; kubectl exec \$p -- ip -br addr; echo; done"

# --- 5. iperf3 inter-node TCP bandwidth baseline ----------------------------
# net-probe-0 = server, net-probe-1 = client. Report single and 8 parallel streams.
SRV_IP="$(kubectl get pod net-probe-0 -o jsonpath='{.status.podIP}')"
echo "iperf3 server IP: $SRV_IP"
kubectl exec net-probe-0 -- bash -c 'iperf3 -s -D; sleep 2'
cap_run $LAB iperf3-1stream.txt  -- kubectl exec net-probe-1 -- iperf3 -c "$SRV_IP" -t 10
cap_run $LAB iperf3-8streams.txt -- kubectl exec net-probe-1 -- iperf3 -c "$SRV_IP" -t 10 -P 8

# --- Teardown probe pods ----------------------------------------------------
kubectl delete pod net-probe-0 net-probe-1 --wait=false

cap_verify_provenance $LAB assets/$LAB "$NODE0,$NODE1" "network-path inspect: gVNIC/TCP, iperf3 baseline (0-GPU)"
echo "lab-05 done."
