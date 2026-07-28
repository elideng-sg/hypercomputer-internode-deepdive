# Lab 22: GPU Fabric Diagnostics — *prove the fabric, or prove it's lying to you*

**Objective:** [lab-18](../lab-18-enable-gpudirect-tcpx/) asks *"can I build the fast
fabric?"*. This lab asks the question support actually gets asked: **"is the fast fabric
working right now, and if not, which layer broke?"** You will stand up a **GPUDirect-TCPXO**
fabric on A3 Mega (one rung above lab-18's TCPX), then run a single command that renders a
PASS/FAIL verdict across all eight layers, validate that command against clusters that are
*deliberately* broken, and package an escalation bundle.

> ### ✅ Status: fabric PROVISIONED & VERIFIED live · throughput number still capacity-gated
> **What was actually done (2026-07-28), not staged:** cluster `hypercomputer-a3-tcpxo`
> (asia-southeast1-c) built with Dataplane V2 + multi-networking + Workload Identity; **8
> GPU VPCs** at MTU 8244; an A3 Mega Flex pool with **8 `--additional-node-network`**
> attachments; 8 `Network`/`GKENetworkParamSet` pairs applied; the NCCL TCPXO installer
> DaemonSet **1/1 Ready**. A real A3 Mega node came up carrying **8 GPU NICs + 1 control
> NIC** with `libnccl-net.so` installed.
>
> **Not claimed:** an enabled-fabric *bandwidth* figure. That needs two A3 Mega nodes up at
> once and Flex returned only one. **Configuration proven ≠ throughput measured** — the
> distinction is the whole point of §5.

**Companion docs:** [doc-25 fabric diagnostics playbook](../../docs/part5-operations-diagnostics/25-fabric-diagnostics-playbook.md)
(the reference material this lab exercises) and [doc-21](../../docs/part6-architecture-gcp-integration/21-gke-network-design.md)
(why the fabric is a create-time decision).

---

## 1. Why this lab exists: the fabric fails *open*

Every other layer of the stack fails loudly. A bad image gives `ImagePullBackOff`. A bad
PVC gives `FailedMount`. An OOM gives exit `137`. **The GPU fabric gives you nothing.** Miss
any layer of the GPUDirect stack and NCCL silently selects its TCP socket transport, the
job finishes, exit codes are `0`, and no error, event or alert is emitted anywhere.

| Path | Measured here | |
|---|---|---|
| Intra-node (NVLink) | **~480 GB/s** | unaffected by fabric bugs — *this is why single-node tests look fine* |
| Inter-node, GPUDirect | ~100–200 GB/s class | the point of TCPX/TCPXO |
| Inter-node, silent fallback | **~28.6 GB/s** | **~17× worse**, zero errors |

Both of this project's original production clusters are in the silent-fallback state right
now and have never reported it. That is not a hypothetical: it is §6.

---

## 2. Build the TCPXO fabric (one rung above lab-18)

```bash
scripts/provision_tcpxo_pool.sh up      # VPCs → cluster → pool → CRDs → plugin
scripts/provision_tcpxo_pool.sh verify
scripts/provision_tcpxo_pool.sh down    # reversible; never touches existing clusters
```

TCPXO is **not "TCPX with more NICs"**. Three things change, and getting any of them wrong
fails open:

| | TCPX (A3 High, lab-18) | TCPXO (A3 Mega, this lab) |
|---|---|---|
| GPU NICs | 4 | **8** |
| Plugin image | `nccl-plugin-gpudirecttcpx` | `nccl-plugin-gpudirecttcpx-dev` **in the `gpudirect-tcpxo` repo** (§3.1) |
| rxdm flavour | `tcpgpudmarxd` | tcpxo-daemon / FasTrak rxdm |
| NCCL env family | `NCCL_GPUDIRECTTCPX_*` | **`NCCL_FASTRAK_*`** |
| Good transport line | `NET/GPUDirectTCPX` | `NET/FasTrak` |

Copying the TCPX recipe and only bumping the NIC count is the single most common TCPXO
bring-up mistake.

### Verify the NICs are really there

The node annotation is ground truth — never hand-guess interface names:

```bash
kubectl get node <node> -o jsonpath='{.metadata.annotations.networking\.gke\.io/nic-info}'
```

Real captured output from this lab's node:

```
NIC COUNT: 9
  eth0 10.148.0.17 0000:00:0c.0      <- control plane (NOT a GPU NIC)
  eth1 192.169.0.2 0000:06:00.0      <- GPU NIC 0
  eth2 192.169.1.2 0000:07:00.0
  ...
  eth8 192.169.7.2 0000:8e:00.0      <- GPU NIC 7
```

Note the off-by-one trap: 9 interfaces on an 8-NIC machine. `eth0` is the control NIC.
Report "9 GPU NICs" to support and you invite a wrong-hardware conclusion.

---

## 3. What running it actually taught us (the real failures)

Every item here was hit **live**, on provisioned Flex GPU time. These are the lab.

### 3.1 `Init:ErrImagePull` — the image name is a trap

```
nccl-tcpxo-installer-4d4xd  0/1  Init:ErrImagePull
Back-off pulling ".../gpudirect-tcpxo/nccl-plugin-gpudirecttcpxo:latest"
```

The obvious name **does not exist**. The TCPXO plugin lives in the `gpudirect-tcpxo`
*repository* under the `nccl-plugin-gpudirecttcpx-dev` *image* name — repo says `tcpxo`,
image says `tcpx`, and neither is a typo for the other. There is also no usable `:latest`
contract: tags are `vN.N.N`, alongside `deprecated-public-image-*` and
`no-new-use-public-image-*` prefixes. Enumerate, never guess:

```bash
gcloud artifacts docker images list \
  us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/nccl-plugin-gpudirecttcpx-dev \
  --include-tags --format='value(tags)'
```

Plugin was `v1.0.17` while rxdm `tcpgpudmarxd-dev` was `v1.0.23` — **versioned separately**.

### 3.2 `pause:3.9` not found — the sidecar, not the plugin

After fixing 3.1 the DaemonSet *still* wasn't Ready. The failing image was
`gcr.io/google-containers/pause:3.9`: that legacy path is frozen and has no 3.9 tag. Use
`registry.k8s.io/pause:3.9`.

**The diagnostic lesson matters more than the fix:** the plugin had installed correctly, but
the DaemonSet never reached Ready because of an unrelated keep-alive container. A "plugin
DaemonSet not ready" symptom does **not** imply a plugin problem. Read *which container*
failed before concluding anything.

### 3.3 Flex-start pool creation is rejected outright

```
flex start node pools require autoscaling enabled
flex start node pools don't support reservations
```

`provision_tcpx_pool.sh` had shipped with `--num-nodes=N` and no autoscaling — it **could
never have worked**; the bug was latent because nobody had capacity to run it. Flex pools
need `--enable-autoscaling --total-min-nodes=0 --total-max-nodes=N
--reservation-affinity=none`, and Pending Pods are what drive the scale-up from 0. Both
provisioning scripts are fixed.

### 3.4 Pod CIDR collision — fails *late*, after the VPCs are built

```
Requested CIDR 10.8.0.0/14 for pods is not available in network default
```

Every cluster in the shared VPC needs a distinct pod range. Survey first — this error
arrives *after* the 8 VPCs exist, and the failed cluster keeps the bad CIDR baked in, so it
must be deleted rather than retried:

```bash
gcloud container clusters list --format='value(name,clusterIpv4Cidr)'
```

### 3.5 `FailedScaleUp: GCE out of resources` — not a bug

A genuine zonal stockout. Nothing to fix; leave the holder armed and retry. Distinguishing
"stocked out" from "misconfigured" saves support hours, which is why the verify script
reports `WARN`, not `FAIL`, when a Flex pool sits at 0 nodes (§4).

---

## 4. The verdict command

```bash
CLUSTER=<cluster> ZONE=<zone> scripts/verify_gpu_fabric.sh
CLUSTER=<cluster> ZONE=<zone> NAMESPACE=<ns> POD=<pod> scripts/verify_gpu_fabric.sh  # preferred
```

Read-only. Exit `0` healthy · `1` degraded · `2` undeterminable. It discovers the machine
family, derives the expected tier, and checks the eight layers that must **all** hold.

Real output on this lab's TCPXO cluster:

```
-- machine=a3-megagpu-8g (source: node) accelerator=nvidia-h100-mega-80gb
-- EXPECTED fabric tier for this family: TCPXO (8 GPU NICs)
 [PASS] 1. Dataplane V2      ADVANCED_DATAPATH
 [PASS] 2. Multi-networking  enabled
 [PASS] 3. GPU Network CRDs  8 NetDevice GPU nets (want >= 8; 9 Networks total incl. built-ins)
 [PASS] 4. Pool GPU NICs     8 additional node networks (want 8)
 [PASS] 5. Jumbo MTU 8244    all gpu-net VPCs >= 8244
 [PASS] 6. NCCL plugin DS    installed and ready (1)
 [PASS] 7. Node NICs         8 GPU NICs + 1 control NIC on gke-...-tcpxo-fl-f106ac3b-03hg (want 8)
 tier=TCPXO  pass=6  fail=0  warn=1
```

### Exercise: validate the checker against known-broken clusters

A checker nobody has seen fail is not evidence. Point it at the single-gVNIC production
clusters, where the expected answer is known:

```bash
CLUSTER=hypercomputer-a3-asiasoutheast1 ZONE=asia-southeast1-c scripts/verify_gpu_fabric.sh; echo $?
```

```
 [FAIL] 1. Dataplane V2      '<empty>' (legacy) — GPUDirect IMPOSSIBLE on this cluster
 [FAIL] 2. Multi-networking  '<empty>' — GPU NICs cannot be attached
 [FAIL] 3. GPU Network CRDs  0 NetDevice GPU nets (want 8; 0 total)
 [FAIL] 4. Pool GPU NICs     0 additional node networks (want 8)
 [FAIL] 6. NCCL plugin DS    not found
 [FAIL] 7. Node NICs         only 0 GPU NICs (want 8)
 tier=TCPXO  pass=1  fail=5  warn=1     -> exit 1
```

Correct on all four clusters tested: all-green on TCPX and TCPXO, 5-FAIL on both
single-gVNIC clusters, with the create-time Dataplane V2 gap named as root cause #1.

**Three false-FAIL bugs the negative controls exposed** (each would have sent support
chasing a non-bug on every stocked-out Flex cluster):
1. Tier undetectable with 0 GPU nodes up → now falls back to the **node pool's** configured
   machine type and reports which source it used.
2. GKE's built-in `default`/`pod-network` Networks inflated the CRD count → now counts only
   `deviceMode: NetDevice`.
3. `FAIL` where `WARN` was correct — 0 ready DaemonSet with **0 GPU nodes** is *expected*,
   not broken.

Fix the checker until its negative controls are right, *then* trust its positives.

---

## 5. The decisive layer — and why §4 isn't enough

Layers 1–7 are **necessary**. Only layer 8 — what transport NCCL actually chose — is
**sufficient**. A cluster can pass all of §4 and still fall back to sockets (wrong env
family, missing rxdm sidecar). **Never close a ticket on configuration alone.**

The workload must run with:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
```

Without both, the transport choice is *never logged* and the ticket cannot be resolved from
logs at all. Then:

```bash
kubectl logs -n <ns> <pod> | grep -E 'NET/(Socket|IB|FasTrak|GPUDirectTCPX)|GPU Direct RDMA'
```

| Line | Verdict |
|---|---|
| `NET/FasTrak` (TCPXO) / `NET/GPUDirectTCPX` (TCPX) / `NET/IB` | ✅ engaged — slowness is *above* the fabric |
| `NET/Socket` | ❌ silent fallback — this is the bug |
| `GPU Direct RDMA Disabled` | ❌ rxdm sidecar missing from the **workload** Pod |
| nothing | ⚠️ debug env not set — no evidence exists yet |

> **Flex-start warning.** When the pool scales back to 0 the node takes its NIC list,
> `dmesg` and NCCL logs with it. **Capture while the node is alive.**

This is precisely why this lab claims a *provisioned, verified* fabric and **not** a
throughput number: the two-node NCCL run needs concurrent Flex capacity that a stockout
denied. Configuration proven, bandwidth unmeasured, and the write-up says so.

---

## 6. Escalation in one command

```bash
NAMESPACE=<ns> POD=<pod> scripts/collect_fabric_bundle.sh
```

Produces `fabric-bundle-<cluster>-<utc>.tar.gz` + a paste-ready `SUMMARY.txt`: the verdict,
create-time gates, per-pool network config, CRDs, node NIC annotations, installer logs,
workload logs (current *and* previous), the in-pod NIC/plugin/`nvidia-smi topo` view, and a
**pre-computed transport verdict** so the reader doesn't have to grep. Validated live
against a running vLLM pod on the degraded mega cluster (21 artefacts collected).

The expensive part of a fabric escalation isn't analysis — it's three days of "please also
send us X". Contents are internal-ish (node names, internal IPs, VPC names, NCCL logs); no
Secrets or workload data are read. Review `SUMMARY.txt` before sending externally.

---

## 7. Monitoring: make a silent failure loud

There is no metric named "NCCL fell back to sockets", so alert on the *shape*: **GPUs busy,
GPU NICs idle.** Full PromQL, alert rules and the suggested Grafana row are in
[doc-25 §5](../../docs/part5-operations-diagnostics/25-fabric-diagnostics-playbook.md#5-monitoring--catching-silent-fallback-before-a-human-notices).
The two that matter:

- `GPUFabricSilentFallbackSuspected` — `DCGM_FI_DEV_GPU_UTIL > 80` **and**
  GPU-NIC (`eth[1-8]`) TX rate near zero for 15m. The only automated warning you will ever
  get for a 17× regression.
- `GPUNCCLPluginNotReady` — plugin DaemonSet ready count 0 **while GPU nodes exist**. The
  second clause is what keeps stocked-out Flex pools from paging anyone (§3.5).

Also track **rail balance**: `max/avg` of per-NIC rates sustained above ~2 during a
collective-heavy phase means traffic is concentrating on one rail instead of spreading.

---

## 8. What support should take from this lab

1. **"No errors" is not "no problem."** The fabric's failure mode is a number, not a log line.
2. **"Job succeeded" proves nothing** about the fabric. Nor does a single-node benchmark —
   NVLink is unaffected by fabric bugs.
3. **Layers 1, 2 and 4 are create-time-only.** If they fail, say "new cluster / recreate the
   pool" on day one rather than debugging layers 5–8 for a week.
4. **Only the NCCL transport line closes a ticket.** Everything else is circumstantial.
5. **Distinguish stocked-out from misconfigured** — a Flex pool at 0 nodes is not a bug.
6. **Read which container failed**, not just which Pod (§3.2).
7. **Validate your tools against known-broken systems** before trusting them on unknown ones (§4).
