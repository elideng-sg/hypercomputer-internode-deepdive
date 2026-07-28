#!/usr/bin/env bash
# provision_tcpxo_pool.sh — stand up a GPUDirect-TCPXO fabric on A3 Mega (reversible).
#
# Sibling of provision_tcpx_pool.sh. Same create-time gates, one rung up the ladder:
#
#   TCPX  (A3 High, a3-highgpu-8g) → 4 GPU NICs, plugin nccl-plugin-gpudirecttcpx
#   TCPXO (A3 Mega, a3-megagpu-8g) → 8 GPU NICs, plugin nccl-plugin-gpudirecttcpxo
#
# Creates (all NEW — never touches the existing hypercomputer-a3-* clusters or their holders):
#   - 8 dedicated GPU VPCs + subnets, jumbo MTU 8244 (one per A3 Mega GPU NIC)
#   - firewall rules permitting intra-VPC TCP/UDP/ICMP on the GPU nets
#   - a GKE cluster with Dataplane V2 + multi-networking + Workload Identity (all create-time only)
#   - an A3 Mega Flex-start node pool with the 8 GPU networks attached
#   - the Network/GKENetworkParamSet CRDs and the nccl-tcpxo-installer DaemonSet
#
# Why a new cluster at all: multi-networking requires Dataplane V2, and BOTH are
# create-time-only settings (gotcha G17). The existing asia-southeast1 cluster has
# `networkConfig.datapathProvider` empty, so TCPXO cannot be added to it in place.
#
# Why Workload Identity here (unlike the TCPX sibling): the existing cluster has WI
# disabled AND restricted node OAuth scopes, so in-pod `gcloud storage` fails with
# "Provided scope(s) are not authorized" — an IAM grant cannot fix a scope limit.
# Enabling WI at create time keeps GCSFuse/GCS access working for the diagnostics
# capture + escalation-bundle upload.
#
# Usage: scripts/provision_tcpxo_pool.sh {up|verify|down}

set -uo pipefail

PROJECT="${PROJECT:-hdlab-elideng}"
REGION="${REGION:-asia-southeast1}"
ZONE="${ZONE:-asia-southeast1-c}"                    # A3 Mega capacity co-located with existing pool
CLUSTER="${CLUSTER:-hypercomputer-a3-tcpxo}"         # NEW cluster
POOL="${POOL:-a3-mega-tcpxo-flex-pool}"
NUM_NODES="${NUM_NODES:-2}"                          # >=2 to show inter-node; cap-of-3 stop-line
MACHINE="${MACHINE:-a3-megagpu-8g}"
ACCEL="${ACCEL:-nvidia-h100-mega-80gb}"
NUM_GPU_NETS="${NUM_GPU_NETS:-8}"                    # A3 Mega exposes 8 GPU NICs (vs High's 4)
GVNIC_MTU="${GVNIC_MTU:-8244}"
PREFIX="${PREFIX:-tcpxo}"
# Pod CIDR must not collide with ANY other cluster sharing the `default` VPC. The bgb-*
# grab clusters occupy 10.8/10.12/10.28/10.68/10.76/10.80/10.84/10.92, and the a3 clusters
# 10.4 (tcpx) /10.32/10.36/10.48/10.100 — so 10.20.0.0/14 is the free window. A collision
# fails the cluster create late, after the VPCs are already built:
#   "Requested CIDR ... is not available in network default"
# Check with: gcloud container clusters list --format='value(name,clusterIpv4Cidr)'
POD_RANGE="${POD_RANGE:-10.20.0.0/14}"
SVC_RANGE="${SVC_RANGE:-10.0.48.0/20}"
CIDR_BASE="${CIDR_BASE:-192.169}"                    # distinct from tcpx's 192.168.x

CRD_MANIFEST="${CRD_MANIFEST:-manifests/tcpxo/network-crds.yaml}"
NCCL_INSTALLER_MANIFEST="${NCCL_INSTALLER_MANIFEST:-manifests/tcpxo/nccl-tcpxo-installer.yaml}"

log(){ echo "[$(basename "$0")] $*"; }
die(){ echo "[$(basename "$0")] ERROR: $*" >&2; exit 1; }

net_idx(){ seq 0 $(( NUM_GPU_NETS - 1 )); }
gpu_net(){  echo "${PREFIX}-gpu-net-$1"; }
gpu_sub(){  echo "${PREFIX}-gpu-sub-$1"; }
gpu_cidr(){ echo "${CIDR_BASE}.$(( $1 )).0/24"; }

create_networks(){
  for i in $(net_idx); do
    local net sub cidr
    net="$(gpu_net "$i")"; sub="$(gpu_sub "$i")"; cidr="$(gpu_cidr "$i")"
    if gcloud compute networks describe "$net" --project "$PROJECT" >/dev/null 2>&1; then
      log "network $net exists — skipping"
    else
      log "creating VPC $net (custom subnet mode, MTU $GVNIC_MTU)"
      gcloud compute networks create "$net" --project "$PROJECT" \
        --subnet-mode=custom --mtu="$GVNIC_MTU" || die "network $net create failed"
    fi
    if gcloud compute networks subnets describe "$sub" --project "$PROJECT" --region "$REGION" >/dev/null 2>&1; then
      log "subnet $sub exists — skipping"
    else
      log "creating subnet $sub ($cidr) in $REGION"
      gcloud compute networks subnets create "$sub" --project "$PROJECT" \
        --network="$net" --region="$REGION" --range="$cidr" || die "subnet $sub create failed"
    fi
    # intra-VPC firewall: allow all between GPU-net hosts (TCPXO data plane)
    gcloud compute firewall-rules create "${net}-internal" --project "$PROJECT" \
      --network="$net" --allow=tcp,udp,icmp --source-ranges="$cidr" 2>/dev/null \
      || log "firewall ${net}-internal exists — skipping"
  done
}

create_cluster(){
  if gcloud container clusters describe "$CLUSTER" --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1; then
    log "cluster $CLUSTER exists — skipping create"; return 0
  fi
  log "creating GKE cluster $CLUSTER (Dataplane V2 + multi-networking + Workload Identity)"
  gcloud container clusters create "$CLUSTER" --project "$PROJECT" --zone "$ZONE" \
    --enable-dataplane-v2 --enable-multi-networking --enable-ip-alias \
    --workload-pool="${PROJECT}.svc.id.goog" \
    --cluster-ipv4-cidr="$POD_RANGE" --services-ipv4-cidr="$SVC_RANGE" \
    --release-channel=regular --num-nodes=1 --machine-type=e2-standard-4 \
    || die "cluster create failed"
}

create_pool(){
  gcloud container clusters get-credentials "$CLUSTER" --project "$PROJECT" --zone "$ZONE" || die "get-credentials failed"
  if gcloud container node-pools describe "$POOL" --project "$PROJECT" --cluster "$CLUSTER" --zone "$ZONE" >/dev/null 2>&1; then
    log "node pool $POOL exists — skipping"; return 0
  fi
  local addl=""
  for i in $(net_idx); do
    addl+=" --additional-node-network=network=$(gpu_net "$i"),subnetwork=$(gpu_sub "$i")"
  done
  log "creating A3 Mega TCPXO node pool $POOL (0→$NUM_NODES × $MACHINE, Flex-start, +${NUM_GPU_NETS} GPU networks)"
  # Flex-start REQUIRES autoscaling and REJECTS reservation affinity (gotcha G19):
  #   "flex start node pools require autoscaling enabled"
  #   "flex start node pools don't support reservations"
  # So: --enable-autoscaling --num-nodes=0 --total-min-nodes=0 --total-max-nodes=$NUM_NODES,
  # and the holder Deployment's Pending Pods are what actually trigger the flex scale-up.
  # shellcheck disable=SC2086
  gcloud container node-pools create "$POOL" --project "$PROJECT" --cluster "$CLUSTER" --zone "$ZONE" \
    --machine-type="$MACHINE" --accelerator=type="$ACCEL",count=8,gpu-driver-version=latest \
    --flex-start --enable-gvnic \
    --enable-autoscaling --num-nodes=0 --total-min-nodes=0 --total-max-nodes="$NUM_NODES" \
    --location-policy=ANY --reservation-affinity=none \
    $addl \
    || die "node pool create failed (likely A3 Mega Flex capacity — see NOTES; fall back to staged)"
}

install_plugin(){
  log "applying Network/GKENetworkParamSet CRDs (${NUM_GPU_NETS} GPU nets)"
  kubectl apply -f "$CRD_MANIFEST" || die "CRD apply failed"
  log "installing nccl-tcpxo-installer DaemonSet"
  kubectl apply -f "$NCCL_INSTALLER_MANIFEST" || die "installer apply failed"
}

verify(){
  gcloud container clusters get-credentials "$CLUSTER" --project "$PROJECT" --zone "$ZONE" 2>/dev/null
  echo "=== create-time gates (must be ADVANCED_DATAPATH + multi-networking) ==="
  gcloud container clusters describe "$CLUSTER" --project "$PROJECT" --zone "$ZONE" \
    --format='value(networkConfig.datapathProvider,networkConfig.enableMultiNetworking,workloadIdentityConfig.workloadPool)'
  echo "=== node pool ==="
  gcloud container node-pools describe "$POOL" --project "$PROJECT" --cluster "$CLUSTER" --zone "$ZONE" \
    --format='value(config.machineType,initialNodeCount,config.flexStart)' 2>/dev/null || echo "pool absent (capacity pending)"
  echo "=== GPU networks attached per node (expect ${NUM_GPU_NETS} + nic0) ==="
  kubectl get nodes -o json 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
for n in d.get("items",[]):
    a=n["metadata"].get("annotations",{})
    print(n["metadata"]["name"], "networks=", a.get("networking.gke.io/nic-info","<none>")[:200])
' 2>/dev/null
  echo "=== installer DaemonSet ==="
  kubectl get ds -n kube-system 2>/dev/null | grep -i tcpxo || echo "installer not present"
}

teardown(){
  log "tearing down (reverse order); existing hypercomputer-a3-* clusters are never touched"
  gcloud container node-pools delete "$POOL" --project "$PROJECT" --cluster "$CLUSTER" --zone "$ZONE" -q 2>/dev/null || true
  gcloud container clusters delete "$CLUSTER" --project "$PROJECT" --zone "$ZONE" -q 2>/dev/null || true
  for i in $(net_idx); do
    gcloud compute firewall-rules delete "$(gpu_net "$i")-internal" --project "$PROJECT" -q 2>/dev/null || true
    gcloud compute networks subnets delete "$(gpu_sub "$i")" --project "$PROJECT" --region "$REGION" -q 2>/dev/null || true
    gcloud compute networks delete "$(gpu_net "$i")" --project "$PROJECT" -q 2>/dev/null || true
  done
  log "teardown complete"
}

case "${1:-up}" in
  up)     create_networks; create_cluster; create_pool; install_plugin; verify ;;
  verify) verify ;;
  down)   teardown ;;
  *) die "usage: $0 {up|verify|down}" ;;
esac

# NOTES
# - A3 Mega Flex capacity: if `node-pools create` errors with a stockout, the zone has no
#   A3 Mega capacity for extra nodes right now. Retry later; do NOT shrink the existing
#   holders to free capacity (memory: always-hold-gpu-after-work).
# - cap-of-3 stop-line: keep NUM_NODES <= 3 on any A3 Flex pool (memory: a3-flex-stop-line).
# - TCPXO vs TCPX is NOT just "more NICs": the plugin, the rxdm daemon flavour, and the
#   NCCL_FASTRAK_* / NCCL_GPUDIRECTTCPX_* env families differ. Re-pull the GKE-published
#   recipe at run time — the image tags move between GKE minor versions.
# - GB200 (a4x-highgpu-4g) cannot use Flex-start at all (workload-policy/topology catch-22),
#   so this pattern does not extend to A4X; use on-demand or Calendar reservations there.
