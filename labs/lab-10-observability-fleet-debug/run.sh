#!/usr/bin/env bash
# lab-10: fleet observability + fault-injection debugging.
# Part A (metrics) is read-only. Part B injects three REVERSIBLE faults, each on
# hhp6's FREE GPUs only (<=4 of 6), never touching the DWS holder on hv7m or vLLM.
# Every fault scenario is applied, captured, and deleted within the script.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib_capture.sh
LAB=lab-10
NS=default

echo "############ Part A — the GKE-managed metrics pipeline (read-only) ############"
cap_run $LAB managed-stack.txt -- bash -c '
  echo "### dcgm-exporter DaemonSet (GKE-managed)"; kubectl get ds -n gke-managed-system dcgm-exporter -o wide
  echo; echo "### dcgm-exporter pods"; kubectl get pods -n gke-managed-system -o wide | grep -E "NAME|dcgm"
  echo; echo "### GMP collectors + operator"; kubectl get pods -n gmp-system -o wide
  echo; echo "### scrape config CRs"; kubectl get clusterpodmonitoring,podmonitoring -A | grep -iE "NAME|dcgm|gpu"'

# live DCGM scrape from the exporter on the vLLM node (real nonzero metrics)
POD=$(kubectl get pods -n gke-managed-system -l app.kubernetes.io/name=gke-managed-dcgm-exporter \
        -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.spec.nodeName}{"\n"}{end}' | grep hhp6 | awk '{print $1}')
echo "scraping dcgm-exporter pod $POD (hhp6) ..."
kubectl port-forward -n gke-managed-system "pod/$POD" 19400:9400 >/dev/null 2>&1 &
PF=$!; sleep 3
cap_run $LAB dcgm-metrics-raw.txt -- bash -c 'curl -s localhost:19400/metrics'
kill $PF 2>/dev/null
cap_run $LAB scrape-config.txt -- bash -c 'kubectl get clusterpodmonitoring gke-managed-dcgm-exporter -o yaml | grep -vE "resourceVersion|uid:|generation|creationTimestamp|managedFields|selfLink"'

echo "############ Part B — fault injection (reversible; hhp6 free GPUs only) ############"

echo "### Fault (a) sig#1: JobSet fail-fast blast radius"
kubectl apply -f manifests/fault-kill-rank.yaml
kubectl wait --for=jsonpath='{.status.replicatedJobsStatus[0].active}'=4 jobset/fault-kill-rank --timeout=120s 2>/dev/null || sleep 40
sleep 25   # let a few healthy iterations run
kubectl delete pod -l job-name=fault-kill-rank-worker-3 --grace-period=0 --force 2>/dev/null
sleep 8
cap_run $LAB fault-kill-rank-jobset-signature.txt -- bash -c '
  kubectl get jobs -l jobset.sigs.k8s.io/jobset-name=fault-kill-rank
  kubectl get jobset fault-kill-rank -o jsonpath="{.status}" | python3 -m json.tool'
kubectl delete jobset fault-kill-rank --ignore-not-found --wait=false

echo "### Fault (a) sig#2: NCCL watchdog abort on surviving ranks (raw pods)"
kubectl apply -f manifests/fault-nccl-hang.yaml
for i in $(seq 1 30); do [ "$(kubectl get pods -l app=nccl-hang --no-headers|grep -c ' Running')" = 4 ] && break; sleep 5; done
kubectl logs -f nccl-hang-0 > "$ASSETS/$LAB/_hang-rank0.log" 2>&1 &
LOGPID=$!; sleep 25
kubectl delete pod nccl-hang-3 --grace-period=0 --force 2>/dev/null
sleep 20; kill $LOGPID 2>/dev/null
# (the committed fault-nccl-hang-signature.txt is the curated view of _hang-rank0.log)
kubectl delete -f manifests/fault-nccl-hang.yaml --ignore-not-found --wait=false

echo "### Fault (c): mismatched-collective config"
kubectl apply -f manifests/fault-env-mismatch.yaml
for i in $(seq 1 24); do [ "$(kubectl get pods -l app=nccl-mismatch --no-headers|grep -c ' Running')" = 2 ] && break; sleep 5; done
sleep 80
cap_run $LAB _mismatch-rank0.txt -- bash -c 'kubectl logs nccl-mismatch-0 | tail -20'
cap_run $LAB _mismatch-rank1.txt -- bash -c 'kubectl logs nccl-mismatch-1 | tail -20'
cap_run $LAB _mismatch-phases.txt -- bash -c 'kubectl get pods -l app=nccl-mismatch -o custom-columns=NAME:.metadata.name,PHASE:.status.phase,EXIT:.status.containerStatuses[0].state.terminated.exitCode'
kubectl delete -f manifests/fault-env-mismatch.yaml --ignore-not-found --wait=false

echo "### Fault (b): NIC saturation steals inter-node bandwidth (hostNetwork, 0 GPU)"
# see run notes; uses two hostNetwork iperf3 probes on the two GPU nodes.
bash labs/lab-10-observability-fleet-debug/nic-saturation.sh || echo "(nic-saturation step skipped)"

cap_verify_provenance $LAB assets/$LAB "hhp6(+hv7m NIC)" "observability: managed DCGM/GMP scrape + 4 fault signatures (kill-rank x2, mismatch, NIC saturation); reversible, holder/vLLM untouched"
echo "lab-10 done. All fault workloads deleted; DWS holder and vLLM untouched."
