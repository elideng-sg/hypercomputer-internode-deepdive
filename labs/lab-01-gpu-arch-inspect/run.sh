#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib_capture.sh"
LAB=lab-01
kubectl apply -f "$REPO_ROOT/scripts/gpu_pod.yaml"
kubectl wait --for=condition=Ready pod/gpu-debug --timeout=300s
cap_run $LAB smi.txt          -- kubectl exec gpu-debug -- nvidia-smi
cap_run $LAB smi-q.txt        -- kubectl exec gpu-debug -- nvidia-smi -q
cap_run $LAB topo.txt         -- kubectl exec gpu-debug -- nvidia-smi topo -m
cap_run $LAB devquery.txt     -- kubectl exec gpu-debug -- bash -lc 'deviceQuery || /usr/local/cuda/extras/demo_suite/deviceQuery || echo "deviceQuery not present"'
cap_verify_provenance $LAB assets/$LAB "$(kubectl get pod gpu-debug -o jsonpath='{.spec.nodeName}')" "arch inspect"
