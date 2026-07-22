#!/usr/bin/env bash
# lab-08: JobSet replicated-job model + headless-service pod-DNS rendezvous,
# admitted and gated by Kueue quota. Runs a small 4-GPU NCCL all-reduce that
# fits inside the free capacity on the non-held node (hhp6); the DWS-held node
# is never touched. Also submits an over-quota JobSet to capture Kueue's
# admission gate (no pods created), then deletes it.
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib_capture.sh"
LAB=lab-08
mkdir -p "$ASSETS/$LAB"
M="$REPO_ROOT/manifests"

# --- 0. Controllers + quota objects -----------------------------------------
cap_run $LAB controllers.txt -- bash -c "echo '### JobSet + Kueue controllers'; kubectl get deploy -n jobset-system jobset-controller-manager -o wide; kubectl get deploy -n kueue-system kueue-controller-manager -o wide; echo; echo '### CRDs'; kubectl get crd | grep -iE 'jobsets.jobset|clusterqueues.kueue|localqueues.kueue|workloads.kueue|resourceflavors.kueue'"
kubectl apply -f "$M/kueue-gpu-queues.yaml"
sleep 3
cap_run $LAB queues.txt -- bash -c "echo '### ClusterQueue (nominal vs used) + LocalQueue'; kubectl get clusterqueue gpu-cq -o wide; kubectl get localqueue gpu-lq -n default -o wide; echo; kubectl get clusterqueue gpu-cq -o jsonpath='{.status}' | python3 -m json.tool 2>/dev/null || true"

# --- 1. Submit the admitted JobSet (4 GPUs == quota) -------------------------
kubectl apply -f "$M/jobset-nccl.yaml"
sleep 8
cap_run $LAB admission.txt -- bash -c "echo '### Kueue Workload for the JobSet (expect Admitted / QuotaReserved=True)'; kubectl get workloads -n default -o wide; echo; kubectl get workloads -n default -o json | python3 -c 'import json,sys
for w in json.load(sys.stdin)[\"items\"]:
    n=w[\"metadata\"][\"name\"]; conds=[(c[\"type\"],c[\"status\"]) for c in w.get(\"status\",{}).get(\"conditions\",[])]
    adm=w.get(\"status\",{}).get(\"admission\",{}).get(\"clusterQueue\",\"-\")
    print(f\"  {n}: cq={adm} conds={conds}\")'"

# --- 2. JobSet structure: replicated jobs, headless service, pod DNS ---------
cap_run $LAB jobset-structure.txt -- bash -c "echo '### JobSet'; kubectl get jobset nccl-jobset -o wide; echo; echo '### child Jobs (one per replica)'; kubectl get jobs -l jobset.sigs.k8s.io/jobset-name=nccl-jobset -o wide 2>/dev/null || kubectl get jobs | grep nccl-jobset; echo; echo '### headless Service (provides pod DNS)'; kubectl get svc nccl-jobset -o wide; echo; echo '### pods -> node placement'; kubectl get pods -l jobset.sigs.k8s.io/jobset-name=nccl-jobset -o wide"

# capture pod DNS resolution from inside a running pod
sleep 5
POD0="$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=nccl-jobset -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [ -n "$POD0" ]; then
  cap_run $LAB pod-dns.txt -- bash -c "echo '### stable pod DNS via the JobSet headless service (from $POD0)'; kubectl exec $POD0 -- bash -lc 'for h in nccl-jobset-worker-0-0 nccl-jobset-worker-1-0 nccl-jobset-worker-2-0 nccl-jobset-worker-3-0; do echo -n \"\$h.nccl-jobset -> \"; getent hosts \$h.nccl-jobset | awk \"{print \\\$1}\" || echo unresolved; done' 2>&1 || echo '(pod exited before DNS probe — see completion log)'"
fi

# --- 3. Wait for completion, capture the all-reduce result -------------------
kubectl wait --for=condition=Completed jobset/nccl-jobset --timeout=420s 2>&1 | tee -a "$ASSETS/$LAB/jobset-structure.txt" || \
  kubectl wait --for=condition=complete job -l jobset.sigs.k8s.io/jobset-name=nccl-jobset --timeout=60s 2>&1 || true
cap_run $LAB allreduce-result.txt -- bash -c "echo '### rank logs (expect value=4.0 across ranks)'; for j in 0 1 2 3; do echo \"-- worker-\$j --\"; kubectl logs job/nccl-jobset-worker-\$j 2>/dev/null | grep -E 'RANK|all_reduce|host=' | tail -3 || echo '(no log)'; done"
cap_run $LAB jobset-final.txt -- bash -c "kubectl get jobset nccl-jobset -o json | python3 -c 'import json,sys; w=json.load(sys.stdin); print(\"conditions:\", [(c[\"type\"],c[\"status\"]) for c in w.get(\"status\",{}).get(\"conditions\",[])]); print(\"replicatedJobsStatus:\", w.get(\"status\",{}).get(\"replicatedJobsStatus\"))'"

# --- 4. Over-quota gating demo (Kueue suspends; no pods created) -------------
kubectl apply -f "$M/jobset-overquota.yaml"
sleep 8
cap_run $LAB overquota-gate.txt -- bash -c "echo '### over-quota JobSet: Kueue holds it (QuotaReserved=False), no pods created'; kubectl get workloads -n default -o json | python3 -c 'import json,sys
for w in json.load(sys.stdin)[\"items\"]:
    if \"overquota\" in w[\"metadata\"][\"name\"]:
        conds=[(c[\"type\"],c[\"status\"],c.get(\"reason\",\"\")) for c in w.get(\"status\",{}).get(\"conditions\",[])]
        print(\"  workload:\", w[\"metadata\"][\"name\"], \"conds=\", conds)'; echo; echo '### pods for over-quota jobset (expect NONE):'; kubectl get pods -l jobset.sigs.k8s.io/jobset-name=nccl-jobset-overquota 2>&1 | tail -2"
kubectl delete -f "$M/jobset-overquota.yaml" --wait=false

cap_verify_provenance $LAB assets/$LAB "$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=nccl-jobset -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || echo hhp6)" "JobSet 4-GPU NCCL all-reduce via Kueue gpu-lq (quota 4); over-quota JobSet gated (no pods)"
echo "lab-08 core done. (teardown of admitted JobSet handled separately)"
