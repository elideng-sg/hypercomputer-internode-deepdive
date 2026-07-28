#!/usr/bin/env bash
# verify_gpu_fabric.sh — one-command GPU-fabric health check for GKE A3/A4 clusters.
#
# AUDIENCE: local and remote support teams. Run this FIRST on any "distributed training is
# slow / hanging / not using the fast network" ticket. It needs no prior context about the
# cluster: it discovers the machine family, works out which fabric rung SHOULD be available,
# checks each of the 8 layers that must all be true for GPUDirect to actually engage, and
# prints a PASS/FAIL table plus the single most likely root cause.
#
# WHY THIS EXISTS: the GPU fabric FAILS OPEN. If any layer below is missing, NCCL silently
# falls back to the TCP socket transport and the job still completes — just ~17x slower
# inter-node (measured: ~480 GB/s intra-node vs ~28.6 GB/s inter-node on single-gVNIC).
# There is no error, no event, no crash. A silent 17x regression is the single most
# expensive failure mode on this platform, so it needs an explicit check.
#
# USAGE:
#   scripts/verify_gpu_fabric.sh                              # current kubectl context
#   CLUSTER=my-cluster ZONE=us-central1-a scripts/verify_gpu_fabric.sh
#   NAMESPACE=training POD=trainer-0 scripts/verify_gpu_fabric.sh   # also inspect a live pod
#
# EXIT CODES:  0 = fabric healthy for its tier   1 = degraded (silent fallback likely)
#              2 = cannot determine (insufficient access)
#
# Read-only. Creates nothing, changes nothing, safe to run against production clusters.

set -uo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
CLUSTER="${CLUSTER:-}"
ZONE="${ZONE:-}"
NAMESPACE="${NAMESPACE:-}"
POD="${POD:-}"

PASS=0; FAIL=0; WARN=0
declare -a ROWS=()
declare -a CAUSES=()

row(){ ROWS+=("$1|$2|$3"); case "$1" in PASS) PASS=$((PASS+1));; FAIL) FAIL=$((FAIL+1));; WARN) WARN=$((WARN+1));; esac; }
cause(){ CAUSES+=("$1"); }
have(){ command -v "$1" >/dev/null 2>&1; }

echo "=============================================================================="
echo " GPU FABRIC HEALTH CHECK"
echo " project=${PROJECT:-<unset>}  cluster=${CLUSTER:-<current-context>}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================================="

have kubectl || { echo "FATAL: kubectl not found"; exit 2; }
have gcloud  || echo "WARN: gcloud not found — cluster-level checks (1,2) will be skipped"

# ---------------------------------------------------------------------------
# Discover the cluster + machine family, which determines the EXPECTED rung.
# ---------------------------------------------------------------------------
if [ -z "$CLUSTER" ]; then
  CTX="$(kubectl config current-context 2>/dev/null)"
  # gke_<project>_<location>_<cluster>
  CLUSTER="$(echo "$CTX" | awk -F_ '{print $NF}')"
  ZONE="${ZONE:-$(echo "$CTX" | awk -F_ '{print $(NF-1)}')}"
fi
echo
echo "-- discovered: cluster=$CLUSTER zone=$ZONE"

MACHINE="$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.node\.kubernetes\.io/instance-type}{"\n"}{end}' 2>/dev/null | grep -E '^a3|^a4' | head -1)"
ACCEL="$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.cloud\.google\.com/gke-accelerator}{"\n"}{end}' 2>/dev/null | grep -v '^$' | head -1)"

# A Flex-start GPU pool sits at 0 nodes whenever capacity is stocked out, so there may be
# NO GPU node to read the family from even on a perfectly-configured cluster. Fall back to
# the node POOL's configured machine type, so the fabric config is still checkable while
# capacity is pending. MACHINE_SRC records which path we used (matters for report honesty).
MACHINE_SRC="node"
if [ -z "$MACHINE" ] && have gcloud && [ -n "$ZONE" ]; then
  MACHINE="$(gcloud container node-pools list --cluster "$CLUSTER" --zone "$ZONE" --project "$PROJECT" \
             --format='value(config.machineType)' 2>/dev/null | grep -E '^a3|^a4' | head -1)"
  [ -n "$MACHINE" ] && MACHINE_SRC="nodepool-config (0 nodes up — Flex capacity pending)"
fi
[ -z "$MACHINE" ] && { MACHINE="<no-gpu-node-or-pool>"; MACHINE_SRC="none"; }

# Expected fabric per machine family. This is the ladder from doc-05/doc-21.
case "$MACHINE" in
  a3-highgpu-8g)  TIER="TCPX";  WANT_NICS=4; PLUGIN_PAT="tcpx";  ENV_FAMILY="NCCL_GPUDIRECTTCPX_";  WANT_TRANSPORT="NET/GPUDirectTCPX" ;;
  a3-megagpu-8g)  TIER="TCPXO"; WANT_NICS=8; PLUGIN_PAT="tcpxo"; ENV_FAMILY="NCCL_FASTRAK_";        WANT_TRANSPORT="NET/FasTrak|NET/GPUDirectTCPX" ;;
  a3-ultragpu-8g|a4-highgpu-8g|a4x-highgpu-4g)
                  TIER="RDMA/RoCE"; WANT_NICS=8; PLUGIN_PAT="none-native"; ENV_FAMILY="NCCL_IB_"; WANT_TRANSPORT="NET/IB" ;;
  *)              TIER="UNKNOWN"; WANT_NICS=0; PLUGIN_PAT=""; ENV_FAMILY=""; WANT_TRANSPORT="" ;;
esac
echo "-- machine=$MACHINE (source: $MACHINE_SRC) accelerator=${ACCEL:-<none>}"
echo "-- EXPECTED fabric tier for this family: $TIER (${WANT_NICS} GPU NICs)"
if [ "$WANT_NICS" -eq 0 ]; then
  echo "-- NOTE: machine family unknown, so NIC-count checks are reported as WARN, not FAIL."
fi
echo

# ---------------------------------------------------------------------------
# LAYER 1 — Dataplane V2. Create-time only. Without it, multi-networking is
# impossible, so GPUDirect can NEVER work on this cluster (gotcha G17).
# ---------------------------------------------------------------------------
if have gcloud && [ -n "$ZONE" ]; then
  DP="$(gcloud container clusters describe "$CLUSTER" --zone "$ZONE" --project "$PROJECT" \
        --format='value(networkConfig.datapathProvider)' 2>/dev/null)"
  if [ "$DP" = "ADVANCED_DATAPATH" ]; then
    row PASS "1. Dataplane V2" "ADVANCED_DATAPATH"
  else
    row FAIL "1. Dataplane V2" "'${DP:-<empty>}' (legacy) — GPUDirect IMPOSSIBLE on this cluster"
    cause "Cluster lacks Dataplane V2. This is a CREATE-TIME-ONLY setting: it cannot be enabled on an existing cluster. A new cluster is required (--enable-dataplane-v2 --enable-multi-networking). This alone caps you at the ~28.6 GB/s single-gVNIC floor."
  fi

  MN="$(gcloud container clusters describe "$CLUSTER" --zone "$ZONE" --project "$PROJECT" \
        --format='value(networkConfig.enableMultiNetworking)' 2>/dev/null)"
  if [ "$MN" = "True" ] || [ "$MN" = "true" ]; then
    row PASS "2. Multi-networking" "enabled"
  else
    row FAIL "2. Multi-networking" "'${MN:-<empty>}' — GPU NICs cannot be attached"
    cause "Multi-networking disabled (create-time only). Without it, Pods cannot attach the dedicated GPU networks."
  fi
else
  row WARN "1-2. Cluster gates" "skipped (need gcloud + ZONE)"
fi

# ---------------------------------------------------------------------------
# LAYER 3 — the Network / GKENetworkParamSet CRDs, one pair per GPU NIC.
# ---------------------------------------------------------------------------
# NOTE: GKE ships built-in Networks named 'default' / 'default-l3' / 'pod-network' on
# multi-networking clusters. Those are NOT GPU networks — counting them makes a cluster
# look configured when it isn't. Count only NetDevice-mode GPU nets.
GPUNETS="$(kubectl get gkenetworkparamsets.networking.gke.io -o json 2>/dev/null \
  | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: print(0); sys.exit(0)
print(len([x for x in d.get("items",[]) if x.get("spec",{}).get("deviceMode")=="NetDevice"]))' 2>/dev/null)"
GPUNETS="${GPUNETS:-0}"
ALLNETS="$(kubectl get networks.networking.gke.io -o name 2>/dev/null | wc -l | tr -d ' ')"
if [ "$WANT_NICS" -eq 0 ]; then
  row WARN "3. GPU Network CRDs" "$GPUNETS NetDevice GPU nets ($ALLNETS total) — expected count unknown"
elif [ "$GPUNETS" -ge "$WANT_NICS" ]; then
  row PASS "3. GPU Network CRDs" "$GPUNETS NetDevice GPU nets (want >= $WANT_NICS; $ALLNETS Networks total incl. built-ins)"
else
  row FAIL "3. GPU Network CRDs" "$GPUNETS NetDevice GPU nets (want $WANT_NICS; $ALLNETS total)"
  cause "Too few NetDevice GPU Networks. Apply manifests/tcpx{,o}/network-crds.yaml — one Network + GKENetworkParamSet per GPU NIC. Do not count GKE's built-in 'default'/'pod-network' Networks: they carry no GPUDirect DMA path."
fi

# ---------------------------------------------------------------------------
# LAYER 4 — the node pool actually has the extra GPU networks attached.
# ---------------------------------------------------------------------------
if have gcloud && [ -n "$ZONE" ]; then
  POOLNETS=0
  for p in $(gcloud container node-pools list --cluster "$CLUSTER" --zone "$ZONE" --project "$PROJECT" \
             --format='value(name)' 2>/dev/null); do
    n="$(gcloud container node-pools describe "$p" --cluster "$CLUSTER" --zone "$ZONE" --project "$PROJECT" \
         --format='value(networkConfig.additionalNodeNetworkConfigs)' 2>/dev/null | tr ';' '\n' | grep -c 'network' || true)"
    [ "${n:-0}" -gt "$POOLNETS" ] && POOLNETS="$n"
  done
  if [ "$WANT_NICS" -eq 0 ]; then
    row WARN "4. Pool GPU NICs" "$POOLNETS additional node networks — expected count unknown"
  elif [ "${POOLNETS:-0}" -ge "$WANT_NICS" ]; then
    row PASS "4. Pool GPU NICs" "$POOLNETS additional node networks (want $WANT_NICS)"
  else
    row FAIL "4. Pool GPU NICs" "$POOLNETS additional node networks (want $WANT_NICS)"
    cause "Node pool was created without enough --additional-node-network flags. This CANNOT be added to an existing pool — recreate the pool with one --additional-node-network per GPU NIC."
  fi
fi

# ---------------------------------------------------------------------------
# LAYER 5 — jumbo MTU on the GPU VPCs. 8244 is the documented value; 1460
# (the default) silently costs a large fraction of achievable bandwidth.
# ---------------------------------------------------------------------------
if have gcloud; then
  SMALL=""
  for net in $(gcloud compute networks list --project "$PROJECT" --format='value(name)' 2>/dev/null | grep -E 'gpu-net'); do
    mtu="$(gcloud compute networks describe "$net" --project "$PROJECT" --format='value(mtu)' 2>/dev/null)"
    [ "${mtu:-0}" -lt 8244 ] && SMALL="$SMALL $net(mtu=$mtu)"
  done
  if [ -n "$SMALL" ]; then
    row FAIL "5. Jumbo MTU 8244" "undersized:$SMALL"
    cause "GPU VPC MTU < 8244. MTU is set at VPC creation; mismatched MTU between peers also causes hangs/retransmits rather than a clean error."
  else
    row PASS "5. Jumbo MTU 8244" "all gpu-net VPCs >= 8244 (or none present)"
  fi
fi

# ---------------------------------------------------------------------------
# LAYER 6 — the NCCL plugin installer DaemonSet is present AND scheduled.
# A common silent failure: the DaemonSet exists but its nodeSelector names the
# wrong accelerator (h100-80gb vs h100-mega-80gb), so it never lands.
# ---------------------------------------------------------------------------
DS="$(kubectl get ds -A 2>/dev/null | grep -iE "nccl.*(tcpx|tcpxo|fastrak)" || true)"
GPUNODES="$(kubectl get nodes -l cloud.google.com/gke-gpu=true -o name 2>/dev/null | wc -l | tr -d ' ')"
if [ -n "$DS" ]; then
  DSREADY="$(echo "$DS" | awk '{print $5}' | head -1)"   # NUMBER READY column
  if [ "${DSREADY:-0}" -gt 0 ]; then
    row PASS "6. NCCL plugin DS" "installed and ready ($DSREADY)"
  elif [ "${GPUNODES:-0}" -eq 0 ]; then
    # 0 ready with 0 GPU nodes is CORRECT, not a fault: a DaemonSet has nowhere to land.
    # Reporting FAIL here would send support chasing a non-bug on every stocked-out
    # Flex cluster. Distinguish "misconfigured" from "no capacity yet".
    row WARN "6. NCCL plugin DS" "installed, 0 ready — no GPU nodes up yet (Flex capacity pending)"
  else
    row FAIL "6. NCCL plugin DS" "present but 0 ready while $GPUNODES GPU node(s) exist"
    cause "The NCCL plugin installer DaemonSet is not running despite GPU nodes being present. Check its nodeSelector matches the node's cloud.google.com/gke-accelerator EXACTLY (nvidia-h100-80gb for A3 High vs nvidia-h100-mega-80gb for A3 Mega) — a mismatch means it never schedules and NCCL silently uses sockets."
  fi
  echo "$DS" | grep -qi "$PLUGIN_PAT" || {
    row WARN "6b. Plugin flavour" "installed plugin does not look like '$PLUGIN_PAT' (expected for $TIER)"
    cause "Plugin flavour mismatch: $TIER needs the *${PLUGIN_PAT}* plugin. Using the TCPX plugin on A3 Mega (or vice-versa) fails open to sockets."
  }
else
  row FAIL "6. NCCL plugin DS" "not found"
  cause "No NCCL GPUDirect plugin installer DaemonSet found. Without libnccl-net-*.so on the node, NCCL uses the built-in socket transport — this is the classic silent-fallback cause."
fi

# ---------------------------------------------------------------------------
# LAYER 7 — per-node NIC count actually realised on the VM.
# ---------------------------------------------------------------------------
NODEJSON="$(kubectl get nodes -o json 2>/dev/null)"
if [ -n "$NODEJSON" ] && [ "$WANT_NICS" -gt 0 ]; then
  echo "$NODEJSON" | python3 -c '
import json,sys,os
want=int(os.environ.get("WANT_NICS","0"))
d=json.load(sys.stdin)
worst=None
for n in d.get("items",[]):
    lbl=n["metadata"].get("labels",{})
    if not lbl.get("cloud.google.com/gke-accelerator"): continue
    ann=n["metadata"].get("annotations",{})
    info=ann.get("networking.gke.io/nic-info") or ann.get("networking.gke.io/north-interfaces") or ""
    total=info.count("birthName") or info.count("ipAddress") or 0
    # nic-info lists ALL interfaces: eth0 is the CONTROL-plane NIC, eth1..ethN are the GPU
    # NICs. Reporting the raw total tells support "9 GPU NICs" on an 8-NIC A3 Mega, which
    # then reads as one-more-than-expected and invites a wrong-hardware conclusion.
    # Subtract the control NIC so the number is directly comparable to WANT_NICS.
    cnt=max(0,total-1) if total else 0
    nm=n["metadata"]["name"]
    if worst is None or cnt<worst[1]: worst=(nm,cnt)
if worst is None:
    print("NONODE|0")
else:
    print(f"{worst[0]}|{worst[1]}")
' 2>/dev/null | while IFS='|' read -r nname ncnt; do
    if [ "$nname" = "NONODE" ]; then
      echo "  [WARN] 7. Node NICs           : no GPU node present (capacity pending?) — cannot verify"
    elif [ "${ncnt:-0}" -ge "$WANT_NICS" ]; then
      echo "  [PASS] 7. Node NICs           : $ncnt GPU NICs + 1 control NIC on $nname (want $WANT_NICS)"
    else
      echo "  [FAIL] 7. Node NICs           : only $ncnt GPU NICs on $nname (want $WANT_NICS)"
      echo "         -> node was not created with the GPU networks attached; recreate the pool."
    fi
  done
else
  row WARN "7. Node NICs" "no GPU nodes to inspect"
fi

# ---------------------------------------------------------------------------
# LAYER 8 — THE decisive check: what transport did NCCL actually pick?
# Everything above is necessary; only this is sufficient. If a pod is named,
# read its log; otherwise tell the operator exactly what to grep for.
# ---------------------------------------------------------------------------
if [ -n "$NAMESPACE" ] && [ -n "$POD" ]; then
  LOG="$(kubectl logs -n "$NAMESPACE" "$POD" --tail=4000 2>/dev/null)"
  if echo "$LOG" | grep -qE "$WANT_TRANSPORT"; then
    row PASS "8. NCCL transport" "$(echo "$LOG" | grep -oE "$WANT_TRANSPORT" | head -1) — GPUDirect ENGAGED"
  elif echo "$LOG" | grep -q "NET/Socket"; then
    row FAIL "8. NCCL transport" "NET/Socket — SILENT FALLBACK, running on plain TCP"
    cause "DECISIVE: NCCL chose NET/Socket, not $WANT_TRANSPORT. The job will complete but inter-node collectives run at roughly 1/17th of intra-node bandwidth. Fix the first FAIL above and re-run."
  else
    row WARN "8. NCCL transport" "no NCCL transport line found (set NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,NET)"
  fi
  if echo "$LOG" | grep -qi "GPU Direct RDMA Disabled"; then
    row FAIL "8b. GPUDirect flag" "'GPU Direct RDMA Disabled' in log"
    cause "NCCL reports GPU Direct disabled — usually the rxdm/tcpxo-daemon sidecar is missing from the WORKLOAD pod (it must share the pod netns; a node-level DaemonSet cannot do this)."
  fi
else
  row WARN "8. NCCL transport" "no pod given — rerun with NAMESPACE=<ns> POD=<pod>"
  echo "         The decisive test. In any GPU pod, set NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET and grep:"
  echo "           GOOD: $WANT_TRANSPORT      BAD: 'NET/Socket' or 'GPU Direct RDMA Disabled'"
fi

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
echo
echo "------------------------------------------------------------------------------"
printf " %-6s %-24s %s\n" "RESULT" "LAYER" "DETAIL"
echo "------------------------------------------------------------------------------"
for r in "${ROWS[@]}"; do
  IFS='|' read -r st ly dt <<<"$r"
  printf " [%-4s] %-24s %s\n" "$st" "$ly" "$dt"
done
echo "------------------------------------------------------------------------------"
echo " tier=$TIER  pass=$PASS  fail=$FAIL  warn=$WARN"
echo

if [ "$FAIL" -gt 0 ]; then
  echo "MOST LIKELY ROOT CAUSE(S), in fix order:"
  i=1; for c in "${CAUSES[@]}"; do echo "  $i) $c"; i=$((i+1)); done
  echo
  echo "NOTE: a degraded fabric does NOT produce errors. Jobs succeed, ~17x slower inter-node."
  echo "Escalating? Run scripts/collect_fabric_bundle.sh to package everything support needs."
  exit 1
fi
echo "Fabric healthy for its tier ($TIER). If throughput is still low, the bottleneck is"
echo "above the fabric: check data loading (starved GPU), NCCL_ALGO, or job-level topology."
exit 0
