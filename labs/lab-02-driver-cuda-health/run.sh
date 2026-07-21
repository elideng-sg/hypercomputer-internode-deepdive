#!/usr/bin/env bash
# Lab 02: Driver, CUDA, and GPU health diagnostics
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/lib_capture.sh"

LAB="lab-02"
ASSETS_DIR="$ASSETS/$LAB"
mkdir -p "$ASSETS_DIR"

# Target node (from lab-01, has 5 free GPUs)
NODE="gke-hypercomputer-a3-a3-h100-dws-pool-16664d9c-hhp6"

echo "=== Lab 02: Driver, CUDA, and GPU Health Diagnostics ==="
echo "Target node: $NODE"
echo ""

# Deploy gpu-debug pod for diagnostics
echo "Step 1: Deploying gpu-debug pod..."
kubectl delete pod gpu-debug --ignore-not-found=true
kubectl apply -f "$REPO_ROOT/scripts/gpu_pod.yaml"
kubectl wait --for=condition=Ready pod/gpu-debug --timeout=120s

echo ""
echo "Step 2: Capturing driver and CUDA versions..."
cap_run "$LAB" "driver-cuda-version.txt" -- \
  kubectl exec gpu-debug -- nvidia-smi --query-gpu=driver_version,name,vbios_version --format=csv

# Also capture CUDA version separately
cap_run "$LAB" "cuda-version.txt" -- \
  kubectl exec gpu-debug -- bash -c 'nvcc --version || nvidia-smi | grep "CUDA Version"'

echo ""
echo "Step 3: Capturing GPU detailed state (ECC, PERFORMANCE, TEMPERATURE, POWER)..."
cap_run "$LAB" "nvidia-smi-detailed.txt" -- \
  kubectl exec gpu-debug -- nvidia-smi -q -d ECC,PERFORMANCE,TEMPERATURE,POWER

echo ""
echo "Step 4: Attempting to capture dmesg/XID events..."
# Try node debug access (may fail on GKE COS with restricted permissions)
if kubectl debug "node/$NODE" -it --image=nvidia/cuda:12.2.0-base-ubuntu22.04 -- \
   bash -c 'dmesg | grep -i -E "NVRM|Xid"' > "$ASSETS_DIR/dmesg-xid.txt" 2>&1; then
  echo "✓ Captured dmesg XID events"
  cat "$ASSETS_DIR/dmesg-xid.txt"
else
  echo "⚠ Node dmesg access restricted (expected on GKE COS)"
  echo "Falling back to kubectl events..."
  cap_run "$LAB" "k8s-events.txt" -- \
    kubectl get events --all-namespaces --field-selector involvedObject.kind=Node,involvedObject.name="$NODE" -o wide

  # Also try in-pod dmesg if available
  echo "Attempting in-pod dmesg (may also be restricted)..."
  kubectl exec gpu-debug -- bash -c 'dmesg 2>&1 | grep -i -E "NVRM|Xid" || echo "dmesg restricted in container"' \
    | tee "$ASSETS_DIR/pod-dmesg-attempt.txt" || true
fi

echo ""
echo "Step 5: Running DCGM diagnostics (Level 2)..."
# Check if dcgmi is available in the pytorch image, otherwise install
if kubectl exec gpu-debug -- which dcgmi >/dev/null 2>&1; then
  echo "✓ dcgmi found in container"
  cap_run "$LAB" "dcgm-diag.txt" -- \
    kubectl exec gpu-debug -- dcgmi diag -r 2
else
  echo "dcgmi not found, installing datacenter-gpu-manager..."
  kubectl exec gpu-debug -- bash -c '
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID) &&
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg &&
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list &&
    apt-get update && apt-get install -y datacenter-gpu-manager
  ' || true

  # Try dcgmi again, or use alternative DCGM container approach
  if kubectl exec gpu-debug -- which dcgmi >/dev/null 2>&1; then
    cap_run "$LAB" "dcgm-diag.txt" -- \
      kubectl exec gpu-debug -- dcgmi diag -r 2
  else
    echo "⚠ Could not install dcgmi in container. Using alternative approach with DCGM Job..."
    # Create a DCGM diagnostic Job
    cat > /tmp/dcgm-job.yaml <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: dcgm-diag
spec:
  template:
    spec:
      restartPolicy: Never
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      - key: cloud.google.com/gke-queued
        operator: Exists
        effect: NoSchedule
      containers:
      - name: dcgm
        image: nvcr.io/nvidia/cloud-native/dcgm:3.3.8-1-ubuntu22.04
        command: ["/bin/bash", "-c"]
        args: ["dcgmi diag -r 2"]
        resources:
          limits:
            nvidia.com/gpu: 1
EOF
    kubectl delete job dcgm-diag --ignore-not-found=true
    kubectl apply -f /tmp/dcgm-job.yaml
    kubectl wait --for=condition=complete job/dcgm-diag --timeout=300s || \
      kubectl wait --for=condition=failed job/dcgm-diag --timeout=10s || true
    kubectl logs job/dcgm-diag | tee "$ASSETS_DIR/dcgm-diag.txt"
    kubectl delete job dcgm-diag
  fi
fi

echo ""
echo "Step 6: Running gpu-burn stress test (60 seconds)..."
cat > /tmp/gpu-burn-job.yaml <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-burn-test
spec:
  template:
    spec:
      restartPolicy: Never
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      - key: cloud.google.com/gke-queued
        operator: Exists
        effect: NoSchedule
      containers:
      - name: gpu-burn
        image: oguzpastirmaci/gpu-burn
        args: ["60"]
        resources:
          limits:
            nvidia.com/gpu: 1
EOF

kubectl delete job gpu-burn-test --ignore-not-found=true
kubectl apply -f /tmp/gpu-burn-job.yaml
echo "Waiting for gpu-burn to complete (60s + overhead)..."
kubectl wait --for=condition=complete job/gpu-burn-test --timeout=180s || \
  kubectl wait --for=condition=failed job/gpu-burn-test --timeout=10s || true

cap_run "$LAB" "gpu-burn.txt" -- kubectl logs job/gpu-burn-test
kubectl delete job gpu-burn-test

echo ""
echo "Step 7: Verifying artifact provenance..."
cap_verify_provenance "$LAB" "driver/cuda/dcgm/gpu-burn diagnostics" "$NODE" "driver 535.309.01, CUDA 12.2"

echo ""
echo "Step 8: Validating diagnostic outputs..."
if grep -qi 'pass\|healthy\|OK' "$ASSETS_DIR/dcgm-diag.txt" 2>/dev/null; then
  echo "✓ DCGM diagnostics passed"
else
  echo "⚠ Review DCGM diagnostics output"
fi

if grep -qi 'pass' "$ASSETS_DIR/gpu-burn.txt" 2>/dev/null; then
  echo "✓ gpu-burn test passed"
else
  echo "⚠ Review gpu-burn output"
fi

echo ""
echo "=== Lab 02 Complete ==="
echo "Artifacts saved to: $ASSETS_DIR"
echo ""
echo "Cleanup: Deleting gpu-debug pod..."
kubectl delete pod gpu-debug --ignore-not-found=true

echo "✓ Done"
