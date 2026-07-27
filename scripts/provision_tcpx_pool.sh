#!/usr/bin/env bash
# provision_tcpx_pool.sh — stand up a GPUDirect-TCPX-capable A3 High environment on GKE.
#
# WHY THIS EXISTS (and why it is *staged*, not yet run live):
#   GPUDirect-TCPX on GKE requires the node pool to attach 4 *additional* GPU networks
#   (one per A3 High GPU NIC). Multi-networking on GKE is only available on clusters with
#   **Dataplane V2** (`--enable-dataplane-v2`) AND **multi-networking** (`--enable-multi-networking`),
#   and BOTH are **create-time-only** cluster settings — they cannot be turned on after the
#   fact. The lab's existing `hypercomputer-a3-asiaeast1` cluster was created WITHOUT
#   Dataplane V2 (verified: `networkConfig.datapathProvider` empty, no anetd/cilium DaemonSet),
#   so TCPX cannot be added to it. Enabling the *enabled* fabric therefore means a **new
#   cluster**. This script provisions that new cluster end-to-end, reversibly.
#
#   It also depends on **A3 High (H100) Flex capacity** in the target region. The project has
#   no on-demand H100 quota (the existing 3-node pool is Flex-start/DWS-provisioned); grabbing
#   2 more A3 nodes is capacity-gated. If capacity is unavailable the pool create will queue or
#   fail — see NOTES. Nothing here is destructive to the existing clusters/holders.
#
# WHAT IT BUILDS (the standard GKE GPUDirect-TCPX recipe for a3-highgpu-8g):
#   - 4 dedicated GPU VPCs + subnets, jumbo MTU 8244 (one per GPU NIC)
#   - firewall rules permitting intra-VPC RDMA/TCP on the GPU nets
#   - a GKE Standard cluster with Dataplane V2 + multi-networking
#   - an A3 High node pool with `--additional-node-network` ×4 (Flex-start), holder-safe
#   - the `Network`/`GKENetworkParamSet` CRDs (manifests/tcpx/network-crds.yaml)
#   - the `nccl-tcpx-installer` DaemonSet (manifests/tcpx/nccl-tcpx-installer.yaml)
#
# USAGE:
#   scripts/provision_tcpx_pool.sh up      # create everything (idempotent-ish; safe to re-run)
#   scripts/provision_tcpx_pool.sh verify  # print readiness (pool, DaemonSet, NIC resources)
#   scripts/provision_tcpx_pool.sh down     # tear everything down (reverse order)
#
# All names are prefixed so teardown is exact and the existing clusters are never touched.
set -uo pipefail

# ---- configuration (override via env) ---------------------------------------
PROJECT="${PROJECT:-hdlab-elideng}"
REGION="${REGION:-asia-east1}"
ZONE="${ZONE:-asia-east1-c}"                       # A3 capacity co-located with existing pool
CLUSTER="${CLUSTER:-hypercomputer-a3-tcpx}"        # NEW cluster (does not touch existing ones)
POOL="${POOL:-a3-tcpx-flex-pool}"
NUM_NODES="${NUM_NODES:-2}"                          # >=2 to show inter-node; cap-of-3 stop-line
MACHINE="${MACHINE:-a3-highgpu-8g}"
GVNIC_MTU="${GVNIC_MTU:-8244}"
PREFIX="${PREFIX:-tcpx}"                             # VPC/subnet/fw name prefix
POD_RANGE="${POD_RANGE:-10.4.0.0/14}"
SVC_RANGE="${SVC_RANGE:-10.0.32.0/20}"
# GKE-published TCPX artifacts (pin at run time — versions move):
NCCL_INSTALLER_MANIFEST="${NCCL_INSTALLER_MANIFEST:-manifests/tcpx/nccl-tcpx-installer.yaml}"
CRD_MANIFEST="${CRD_MANIFEST:-manifests/tcpx/network-crds.yaml}"

log(){ echo "[$(basename "$0")] $*"; }
die(){ echo "[$(basename "$0")] ERROR: $*" >&2; exit 1; }

# ---- the 4 GPU networks (one per A3 High GPU NIC) ----------------------------
gpu_net(){  echo "${PREFIX}-gpu-net-$1"; }
gpu_sub(){  echo "${PREFIX}-gpu-sub-$1"; }
gpu_cidr(){ echo "192.168.$(( $1 )).0/24"; }        # 192.168.0/1/2/3.0/24

create_networks(){
  for i in 0 1 2 3; do
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
    # intra-VPC firewall: allow all between GPU-net hosts (RDMA/TCPX data plane)
    gcloud compute firewall-rules create "${net}-internal" --project "$PROJECT" \
      --network="$net" --allow=tcp,udp,icmp --source-ranges="$cidr" 2>/dev/null \
      || log "firewall ${net}-internal exists — skipping"
  done
}

create_cluster(){
  if gcloud container clusters describe "$CLUSTER" --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1; then
    log "cluster $CLUSTER exists — skipping create"; return 0
  fi
  log "creating GKE cluster $CLUSTER (Dataplane V2 + multi-networking; the create-time gates TCPX needs)"
  gcloud container clusters create "$CLUSTER" --project "$PROJECT" --zone "$ZONE" \
    --enable-dataplane-v2 --enable-multi-networking --enable-ip-alias \
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
  for i in 0 1 2 3; do
    addl+=" --additional-node-network=network=$(gpu_net "$i"),subnetwork=$(gpu_sub "$i")"
  done
  log "creating A3 TCPX node pool $POOL ($NUM_NODES × $MACHINE, Flex-start, +4 GPU networks)"
  # Flex-start (queued provisioning) — matches the existing pool's scarce-capacity posture.
  # shellcheck disable=SC2086
  gcloud container node-pools create "$POOL" --project "$PROJECT" --cluster "$CLUSTER" --zone "$ZONE" \
    --machine-type="$MACHINE" --accelerator=type=nvidia-h100-80gb,count=8,gpu-driver-version=latest \
    --num-nodes="$NUM_NODES" --flex-start --enable-gvnic \
    $addl \
    || die "node pool create failed (likely A3 Flex capacity — see NOTES; fall back to staged)"
}

install_plugin(){
  log "applying Network/GKENetworkParamSet CRDs"
  kubectl apply -f "$CRD_MANIFEST" || die "CRD apply failed"
  log "installing nccl-tcpx-installer DaemonSet + tcpx-daemon"
  kubectl apply -f "$NCCL_INSTALLER_MANIFEST" || die "installer apply failed"
}

verify(){
  gcloud container clusters get-credentials "$CLUSTER" --project "$PROJECT" --zone "$ZONE" 2>/dev/null
  echo "=== node pool ==="
  gcloud container node-pools describe "$POOL" --project "$PROJECT" --cluster "$CLUSTER" --zone "$ZONE" \
    --format="value(status,initialNodeCount,config.machineType)" 2>&1
  echo "=== TCPX installer DaemonSet ==="
  kubectl -n kube-system get ds nccl-tcpx-installer 2>&1
  echo "=== GPU-NIC resource on nodes (networking.gke.io/... ) ==="
  kubectl get nodes -o json 2>/dev/null | python3 -c "
import sys,json
for n in json.load(sys.stdin).get('items',[]):
    al=n.get('status',{}).get('allocatable',{})
    nics=[k for k in al if 'networking.gke.io' in k or 'gpu' in k.lower()]
    print(n['metadata']['name'], {k:al[k] for k in nics})
" 2>&1
}

teardown(){
  log "TEARDOWN — reverse order; existing clusters/holders untouched"
  gcloud container node-pools delete "$POOL" --project "$PROJECT" --cluster "$CLUSTER" --zone "$ZONE" -q 2>/dev/null || true
  gcloud container clusters delete "$CLUSTER" --project "$PROJECT" --zone "$ZONE" -q 2>/dev/null || true
  for i in 0 1 2 3; do
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
# - A3 Flex capacity: if `node-pools create` hangs in PROVISIONING or errors with a stockout,
#   the region has no A3 capacity for the extra nodes right now. Retry later or pick another
#   zone with A3 High capacity; do NOT reduce the existing holders to free capacity.
# - cap-of-3 stop-line: keep NUM_NODES <= 3 on any A3 Flex pool (see memory a3-flex-stop-line).
# - Cost/hold: if kept for the scaling enabled-curve cross-run, hold this pool like the others.
# - The nccl-tcpx-installer image + the exact NCCL_* env are GKE-published and version-pinned;
#   re-pull the current recipe at run time (they move between GKE minor versions).
