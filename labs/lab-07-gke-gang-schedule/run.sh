#!/usr/bin/env bash
# lab-07: GKE GPU scheduling & topology. Read-only survey + a Pending gang demo.
# Requests NO GPUs that ever run: the demo Job is submitted only to capture its
# FailedScheduling event, then deleted. Never disturbs holders or running work.
set -euo pipefail
source "$(dirname "$0")/../../scripts/lib_capture.sh"
LAB=lab-07
mkdir -p "$ASSETS/$LAB"
mapfile -t NODES < <(cap_nodes)

# --- 1. The GPU device plugin advertises nvidia.com/gpu ----------------------
cap_run $LAB device-plugin.txt -- bash -c "kubectl get ds -n kube-system -o wide | awk 'NR==1 || /nvidia-gpu-device-plugin/'; echo; echo '### running device-plugin pods:'; kubectl get pods -n kube-system -o wide | grep nvidia-gpu-device-plugin | grep -v '0/'"

# --- 2. Node labels (topology) + taints (gpu + DWS queued) -------------------
cap_run $LAB node-topology.txt -- bash -c "for n in ${NODES[0]} ${NODES[1]}; do echo \"### \$n labels\"; kubectl get node \$n -o json | python3 -c 'import json,sys; l=json.load(sys.stdin)[\"metadata\"][\"labels\"]; [print(\" \",k,\"=\",v) for k,v in sorted(l.items()) if any(t in k for t in [\"topology\",\"accelerator\",\"nodepool\",\"instance-type\",\"provisioning\"])]'; echo \" taints:\"; kubectl get node \$n -o jsonpath='{range .spec.taints[*]}  {.key}={.value}:{.effect}{\"\n\"}{end}'; echo; done"

# --- 3. DWS ProvisioningRequests — the queued-provisioning gang objects -------
cap_run $LAB provisioning-requests.txt -- bash -c "kubectl get provisioningrequests -A; echo; echo '### conditions:'; kubectl get provisioningrequests -A -o json | python3 -c 'import json,sys
for p in json.load(sys.stdin)[\"items\"]:
    n=p[\"metadata\"][\"name\"]; conds=[(c[\"type\"],c[\"status\"]) for c in p.get(\"status\",{}).get(\"conditions\",[])]
    print(f\"  {n}: {conds}\")'"

# --- 4. The live capacity-holder gang state ----------------------------------
cap_run $LAB holder-gang-state.txt -- bash -c "kubectl get pods -o wide | awk 'NR==1 || /holder/'; echo; echo '### interpretation: a2=Running holds one node; b/c=Pending wait on req-zone-b/c (Provisioned=False)'"

# --- 5. Gang demo: submit a 2x8-GPU Job that cannot fit -> capture the gate ---
kubectl apply -f "$REPO_ROOT/manifests/gang-pending-demo.yaml"
sleep 15
cap_run $LAB gang-pending-events.txt -- bash -c "echo '### pods (expect Pending):'; kubectl get pods -l job-name=gang-pending-demo -o wide; echo; echo '### scheduling events:'; kubectl get events --field-selector reason=FailedScheduling --sort-by=.lastTimestamp | tail -5; echo; echo '### describe (one pod):'; POD=\$(kubectl get pods -l job-name=gang-pending-demo -o jsonpath='{.items[0].metadata.name}'); kubectl describe pod \$POD | sed -n '/Events:/,\$p'"
kubectl delete -f "$REPO_ROOT/manifests/gang-pending-demo.yaml" --wait=false

# --- 6. What is actually scheduled where -------------------------------------
cap_run $LAB gpu-pod-placement.txt -- bash -c "kubectl get pods -A -o wide | awk 'NR==1' ; kubectl get pods -A -o json | python3 -c 'import json,sys
for p in json.load(sys.stdin)[\"items\"]:
    g=sum(int(c.get(\"resources\",{}).get(\"limits\",{}).get(\"nvidia.com/gpu\",0)) for c in p[\"spec\"][\"containers\"])
    if g: print(f\"  {p[\"metadata\"][\"namespace\"]}/{p[\"metadata\"][\"name\"]}  gpu={g}  node={p[\"spec\"].get(\"nodeName\",\"-\")}  phase={p[\"status\"][\"phase\"]}\")'"

cap_verify_provenance $LAB assets/$LAB "${NODES[0]},${NODES[1]}" "GKE scheduling/topology: device-plugin, dual taints, DWS ProvisioningRequests, gang Pending gate (no GPUs consumed)"
echo "lab-07 done."
