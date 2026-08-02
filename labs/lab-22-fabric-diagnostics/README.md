# Lab 22: GPU Fabric Diagnostics — *prove the fabric, or prove it's lying to you*

**Objective:** [lab-18](../lab-18-enable-gpudirect-tcpx/) asks *"can I build the fast
fabric?"*. This lab asks the question support actually gets asked: **"is the fast fabric
working right now, and if not, which layer broke?"** You will stand up a **GPUDirect-TCPXO**
fabric on A3 Mega (one rung above lab-18's TCPX), then run a single command that renders a
PASS/FAIL verdict across all nine layers, validate that command against clusters that are
*deliberately* broken, and package an escalation bundle.

> ### ✅ Status: fabric PROVISIONED, VERIFIED **and MEASURED** live
> **What was actually done (2026-07-28), not staged:** cluster `hypercomputer-a3-tcpxo`
> (asia-southeast1-c) built with Dataplane V2 + multi-networking + Workload Identity; **8
> GPU VPCs** at MTU 8244; an A3 Mega Flex pool with **8 `--additional-node-network`**
> attachments; 8 `Network`/`GKENetworkParamSet` pairs applied; the official NCCL TCPXO
> installer + `nri-device-injector` DaemonSets **2/2 Ready**. Two A3 Mega nodes came up, each
> carrying **8 GPU NICs + 1 control NIC**.
>
> **Measured, not inferred:** a 2-node / **16-GPU** all-reduce reached **317.84 GB/s busbw**
> with `NET/FasTrak` on every rank and **zero** `NET/Socket` lines — against **23.7 GB/s** on
> the gVNIC-only 2-node path of [lab-12](../lab-12-scaling-sweep/), i.e. **≈13.4×**.
> That closes the inter-node "cliff" this guide has carried as an open item since
> [lab-06](../lab-06-2node-nccl-collectives/). Evidence:
> [`tcpxo_allreduce_16gpu.txt`](../../assets/lab-22/tcpxo_allreduce_16gpu.txt),
> [`tcpxo_transport_fastrak.txt`](../../assets/lab-22/tcpxo_transport_fastrak.txt),
> [`tcpxo_inpod_fabric.txt`](../../assets/lab-22/tcpxo_inpod_fabric.txt).
>
> **Monitoring validated against live Managed Prometheus (2026-07-29)** by running the same
> job twice — over FasTrak, then forced onto the socket path — and querying GMP during both.
> This **overturned this lab's original alert rule**: GPU-NIC byte counters read *zero* on a
> healthy TCPXO fabric (GPUDirect bypasses the kernel stack), so "GPUs busy + NICs idle"
> pages on health. `DCGM_FI_PROF_PCIE_TX_BYTES` is the signal that actually discriminates —
> **1327 vs 29.7 Gbit/s**, while GPU util is 100% on both sides. See §7 and
> [`tcpxo_monitoring_validation.txt`](../../assets/lab-22/tcpxo_monitoring_validation.txt).
>
> **Retracted 2026-08-02 — there is no "tuned figure" to claim.**
> This lab shipped 317.84 GB/s as an untuned **floor** because the run logged
> `CPU affinity ... not a subset` advisories. [lab-23](../lab-23-enabled-scaling-curve/) went
> looking for the ceiling and found that **14 NCCL variables are `POLICY_ENFORCED`** by the
> plugin's Guest Config Checker shim, which **aborts NCCL init** rather than warning:
> `NCCL_FASTRAK_NUM_FLOWS=4` and `NCCL_MIN_NCHANNELS=8` each killed the job
> ([G34](../../reference/lab-build-gotchas.md)). The vendor profile is a **contract**, not a
> baseline to improve on — so 317.84 GB/s is simply *the* number for this shape, and the
> affinity advisory, while real, is not addressable through the NCCL environment. lab-23 also
> **independently reproduced this measurement at 316.93 GB/s** (0.3%) with a different runner.
>
> **Also do not read it as a scaling prediction:** the same fabric measures **184.03 GB/s at
> 24 GPUs / 3 nodes** — a 42% fall on the third node.
>
> **Update 2026-07-31 — the TCPX tier is measured too** ([lab-18](../lab-18-enable-gpudirect-tcpx/)):
> **83.27 GB/s** busbw on 2 × `a3-highgpu-8g`, `GPUDirectTCPX_v7`, 4 rails, **≈3.5×** the
> gVNIC floor. Two of this lab's conclusions turned out to be **TCPXO-specific** rather than
> GPUDirect-wide — the netdev counter blindness (§7.1) and the "source the vendor env profile"
> rule (§5) — and are scope-corrected below. Quote **3.5×** for A3 High and **13.4×** for A3
> Mega; the two are not interchangeable.

**Companion docs:** [doc-25 fabric diagnostics playbook](../../docs/part5-operations-diagnostics/25-fabric-diagnostics-playbook.md)
(the reference material this lab exercises) and [doc-21](../../docs/part6-architecture-gcp-integration/21-gke-network-design.md)
(why the fabric is a create-time decision).

---

## 1. Why this lab exists: the fabric fails *open*

Every other layer of the stack fails loudly. A bad image gives `ImagePullBackOff`. A bad
PVC gives `FailedMount`. An OOM gives exit `137`. **The GPU fabric gives you nothing.** Miss
any layer of the GPUDirect stack and NCCL silently selects its TCP socket transport, the
job finishes, exit codes are `0`, and no error, event or alert is emitted anywhere.

| Path | Measured | |
|---|---|---|
| Intra-node (NVLink) | **~480 GB/s** | unaffected by fabric bugs — *this is why single-node tests look fine* |
| Inter-node, **GPUDirect-TCPXO** | **317.84 GB/s** | measured here, §5.1 — 16 GPUs, `NET/FasTrak` |
| Inter-node, **GPUDirect-TCPX** | **83.27 GB/s** | measured in [lab-18](../lab-18-enable-gpudirect-tcpx/) — 16 GPUs, `GPUDirectTCPX_v7`, 4 rails |
| Inter-node, silent fallback | **23.7 / ~28.6 GB/s** | **≈13.4×** worse on A3 Mega, **≈3.5×** on A3 High, zero errors either way ([lab-12](../lab-12-scaling-sweep/) / [lab-06](../lab-06-2node-nccl-collectives/)) |

Both of this project's original production clusters are in the silent-fallback state right
now and have never reported it. That is not a hypothetical: it is §6.

The middle row is the reason this lab exists. It is worth **more than an order of
magnitude**, it is invisible in exit codes, and until this lab it was the one number the
whole guide could not put on the table.

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
| Plugin image | `gpudirect-tcpx/nccl-plugin-gpudirecttcpx-dev` — pin **`:v3.1.12`**, never `:latest` (G29) | `nccl-plugin-gpudirecttcpx-dev` **in the `gpudirect-tcpxo` repo** (§3.1) |
| rxdm flavour | `tcpgpudmarxd` | tcpxo-daemon / FasTrak rxdm |
| NCCL env family | `NCCL_GPUDIRECTTCPX_*` | **`NCCL_FASTRAK_*`** |
| Good transport line | `NET/GPUDirectTCPX` (`_v7`) | `NET/FasTrak` |
| Vendor env-profile script | **none ships** — write the rail list by hand (G28) | `nccl-env-profile.sh` — source it, never hand-write |
| `/dev/aperture_devices` | absent (correct) | required, one per GPU NIC |
| Netdev byte counters under load | **~50 GB/rail — usable** | **~zero — blind** (§7.1) |
| Measured busbw, 16 GPU | **83.27 GB/s** (lab-18) | **317.84 GB/s** (§5.1) |

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

The plugin and the rxdm daemon are **versioned separately, and both are pinned to the GPU
driver** — this is the trap that costs the most time, because a mismatch fails *open* at the
device layer (§3.6), not at image pull:

| Node GPU driver | rxdm `tcpgpudmarxd-dev` | plugin `nccl-plugin-gpudirecttcpx-dev` |
|---|---|---|
| **580.65.06+** (R580) — this cluster ran `580.159.04` | **`v1.0.22`** | **`v1.0.16`** |
| 595.71.05+ (R595) | `v1.0.23` | `v1.0.17` |

Read the driver off the node first, then pick the row — never pick "latest":

```bash
kubectl get node <node> -o jsonpath='{.metadata.labels.cloud\.google\.com/gke-gpu-driver-version}'
kubectl get node <node> -o jsonpath='{.status.nodeInfo.kernelVersion}{"\n"}'
```

### 3.2 `pause:3.9` not found — the sidecar, not the plugin

After fixing 3.1 the DaemonSet *still* wasn't Ready. The failing image was
`gcr.io/google-containers/pause:3.9`: that legacy path is frozen and has no 3.9 tag. Use
`registry.k8s.io/pause:3.9`.

> **This recurred on the TCPX cluster three days later** (**G27**, [lab-18](../lab-18-enable-gpudirect-tcpx/))
> — same dead image, same `0 ready` DaemonSet, same red herring. Two working replacements exist
> and either is fine: `registry.k8s.io/pause:3.9` (used here) or
> `gke.gcr.io/pause:3.8@sha256:880e63f9…` (used by the TCPX manifests, pinned by digest so the
> tag can never move under it). The digest form is preferred for anything you hand to a
> customer. What is *not* fine is the image the GKE docs still name.

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

### 3.6 The dmabuf importer failure — a fail-open *one level worse* than §1

This one is the most valuable finding in the lab, because it defeats layers 1–7 completely.
Symptom: the workbench Pod sits at **`1/2` Ready forever**. The workload container is
`Running` and healthy. The rxdm sidecar is `Terminated`, and this is the part that matters:

```
    State:      Terminated
      Reason:   Completed          <-- not Error
      Exit Code: 0                 <-- not 1
```

**Exit `0`, reason `Completed`.** No `CrashLoopBackOff`, no restart count, no Warning event,
no non-zero exit anywhere in `kubectl describe`. Every Kubernetes-level signal says the
sidecar did its job and stopped. Only the sidecar's *own* logs disagree:

```
Failed to create dmabuf importer context
Memory importer: failed to initialize addr translator on NIC IP 192.169.0.3
Exiting with result:1              <-- logs say 1, process exits 0
```

Root cause was **not** a version mismatch (it reproduced identically on rxdm `v1.0.17` *and*
`v1.0.23`). The real cause: a hand-written installer DaemonSet that ran only

```bash
/scripts/container_entry.sh install --install-nccl
```

and **omitted the official installer's `pre-installation` initContainer**, which is where the
two device nodes rxdm needs are actually created:

```bash
nsenter -at 1 -- modprobe import-helper          # creates /dev/dmabuf_import_helper (10:260)
# + one bind-mount per GPU-NIC PCI function under /dev/aperture_devices/<BDF>
```

Without them rxdm cannot register GPU memory with the NIC — so it gives up, and the fabric
is dead. **That installer passed layers 1 through 7 of §4 while the cluster could not move a
single byte over the fabric.** Two lessons:

1. **Never hand-roll the installer.** Apply the official
   [`nccl-tcpxo-installer.yaml`](../../manifests/tcpxo/nccl-tcpxo-installer.yaml) *and*
   [`nri-device-injector.yaml`](../../manifests/tcpxo/nri-device-injector.yaml) — the injector
   is what honours the `devices.gke.io/container.tcpxo-daemon` Pod annotation and puts the
   GPUs + `/dev/dmabuf_import_helper` **inside the sidecar**.
2. Verify the devices directly. Two commands, and the checker now does it for you as
   **layer 6c**:

```bash
ls -l /dev/dmabuf_import_helper       # want: crw------- 1 root root 10, 260
ls /dev/aperture_devices/ | wc -l     # want: 8 on A3 Mega (one per GPU NIC BDF)
```

Full signature, including the version-pairing table and the fix sequence:
[`assets/lab-22/tcpxo_failure_dmabuf_importer.txt`](../../assets/lab-22/tcpxo_failure_dmabuf_importer.txt).

### 3.7 `UnexpectedAdmissionError` — don't pin GPU Pods with `nodeName`

```
UnexpectedAdmissionError: Allocate failed due to requested number of devices
unavailable for nvidia.com/gpu. Requested: 8, Available: 0
```

The workbench had `nodeName: gke-...` to force both ranks onto specific nodes. `nodeName`
**bypasses the scheduler**, so there is no queue: kubelet admits the Pod immediately, finds
the GPUs already allocated (here: to capacity holders), and hard-**rejects** it. You get a
terminal error instead of the `Pending` you wanted.

Use scheduler placement instead — the Pods then wait in line *before* the incumbent is
released, which is what makes a gap-free handover possible at all:

```yaml
nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-h100-mega-80gb
affinity:
  podAntiAffinity:                 # one rank per node
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector: { matchLabels: { app: tcpxo-wb } }
      topologyKey: kubernetes.io/hostname
```

---

## 4. The verdict command

```bash
CLUSTER=<cluster> ZONE=<zone> scripts/verify_gpu_fabric.sh
CLUSTER=<cluster> ZONE=<zone> NAMESPACE=<ns> POD=<pod> scripts/verify_gpu_fabric.sh  # preferred
```

Read-only. Exit `0` healthy · `1` degraded · `2` undeterminable. It discovers the machine
family, derives the expected tier, and checks the nine layers that must **all** hold (1-8 plus **6c**, added after a bring-up that passed 1-7 and still moved no bytes).

Real output on this lab's TCPXO cluster, **after** §3.6 was fixed:

```
-- machine=a3-megagpu-8g (source: node) accelerator=nvidia-h100-mega-80gb
-- EXPECTED fabric tier for this family: TCPXO (8 GPU NICs)
 [PASS] 1. Dataplane V2      ADVANCED_DATAPATH
 [PASS] 2. Multi-networking  enabled
 [PASS] 3. GPU Network CRDs  8 NetDevice GPU nets (want >= 8; 9 Networks total incl. built-ins)
 [PASS] 4. Pool GPU NICs     8 additional node networks (want 8)
 [PASS] 5. Jumbo MTU 8244    all gpu-net VPCs >= 8244
 [PASS] 6. NCCL plugin DS    installed and ready (2)
 [PASS] 6c. rxdm devices     /dev/dmabuf_import_helper present, 8 aperture devices (want 8)
 [PASS] 7. Node NICs         8 GPU NICs + 1 control NIC on gke-...-tcpxo-fl-f106ac3b-03hg (want 8)
 [PASS] 8. NCCL transport    NET/FasTrak — GPUDirect ENGAGED
 tier=TCPXO  pass=8  fail=0  warn=0
```

**`pass=8 fail=0 warn=0` is the only clean sheet in this repo**, and it is the *only* form of
this table that closes a ticket — because it includes layer 8. Anything less is
configuration evidence, not fabric evidence.

**Layer 6c is new, and it exists because of §3.6.** Layers 1–7 as originally written all
passed on a cluster whose fabric could not move a byte; 6c is the layer that catches it,
by probing the node for `/dev/dmabuf_import_helper` and counting
`/dev/aperture_devices/<BDF>` entries against the expected NIC count. It runs on TCPXO only.

> **Implementation note worth stealing.** The obvious way to read a node device is
> `kubectl debug node/<n> -- ls /dev`. Do **not** use it in a script: without a TTY it
> returns *no stdout at all*, which reads as "device missing" and produces a confident
> **false FAIL**. 6c uses an ordinary Pod with `/dev` hostPath-mounted read-only and reads
> `kubectl logs`, which is deterministic — and if the probe returns nothing it emits `WARN`
> ("could not verify"), never `FAIL`. **A checker must not assert a fault it did not
> observe.**

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
 [PASS] 5. Jumbo MTU 8244    all gpu-net VPCs >= 8244 (or none present)
 [FAIL] 6. NCCL plugin DS    not found
 [FAIL] 6c. rxdm devices     /dev/dmabuf_import_helper MISSING on gke-...-a3-mega-flex-poo-44c95a12-5w60
 [FAIL] 7. Node NICs         only 0 GPU NICs (want 8)
 tier=TCPXO  pass=1  fail=6  warn=1     -> exit 1
```

Correct on all four clusters tested: all-green on TCPX and TCPXO, all-red on both
single-gVNIC clusters, with the create-time Dataplane V2 gap named as root cause #1. Note
that 6c fires here **on a live GPU node** — this is the negative control that proves 6c can
distinguish "device absent" from "node absent". Both control runs, verbatim:
[`assets/lab-22/verify_layer8_fix_controls.txt`](../../assets/lab-22/verify_layer8_fix_controls.txt).

**Three false-FAIL bugs the negative controls exposed** (each would have sent support
chasing a non-bug on every stocked-out Flex cluster):
1. Tier undetectable with 0 GPU nodes up → now falls back to the **node pool's** configured
   machine type and reports which source it used.
2. GKE's built-in `default`/`pod-network` Networks inflated the CRD count → now counts only
   `deviceMode: NetDevice`.
3. `FAIL` where `WARN` was correct — 0 ready DaemonSet with **0 GPU nodes** is *expected*,
   not broken.
4. (layer 6c) A silent probe treated as a missing device → now `WARN`, not `FAIL`.

**A fourth, found later on the TCPX cluster (2026-07-31, G30).** Layer 6 reported a
`nodeSelector` mismatch for a DaemonSet whose Pods **were** scheduled — the real fault was a
dead `pause` image, and the plugin was *already installed*. One symptom (`0 ready`), two
unrelated causes, and the checker had picked the wrong one confidently. Layer 6 now counts
scheduled Pods first and names the stuck container; validated against **both** faults injected
live: [`assets/lab-18/checker_bug4_layer6_controls.txt`](../../assets/lab-18/checker_bug4_layer6_controls.txt).
That makes **four** false-attribution bugs found by controls, all of the same species: an
aggregate signal read as a specific cause.

### The two bugs the *positive* control exposed — and why they were worse

Negative controls found false FAILs. Then the first genuinely-healthy cluster found something
worse: **a false "no evidence"** on the decisive layer. The checker printed

```
 [WARN] 8. NCCL transport    no NCCL transport line found (set NCCL_DEBUG=INFO, ...)
```

against a pod whose own log contained **153 `NET/FasTrak` lines**. Two independent defects:

1. **`kubectl logs <pod>` defaults to the *first* container.** On every TCPXO workload pod
   that is the rxdm sidecar, which never prints a transport line. kubectl even says so —
   `Defaulted container "tcpxo-daemon" out of: tcpxo-daemon, bench` — but the script sent
   stderr to `/dev/null`. Per-container counts: `tcpxo-daemon` 0/261 lines, `bench` 153/4000.
   Fix: `--all-containers`.

2. **`set -o pipefail` + `grep -q` = false negative on large input.** This is the subtle one.
   `grep -q` exits `0` on its *first* match without draining stdin; the upstream `echo` then
   takes `SIGPIPE` and dies `141`; under `pipefail` the **pipeline** yields 141, so the `if`
   takes the **false** branch although the pattern matched:

   ```bash
   $ set -uo pipefail
   $ echo "$LOG" | grep -qE "NET/FasTrak"; echo $?
   141                        # "no match" — with 153 matches present
   $ grep -qE "NET/FasTrak" <<<"$LOG"; echo $?
   0                          # correct
   ```

   It is **input-size dependent**: a log that fits the 64 KiB pipe buffer lets `echo` finish
   before `grep` exits, and the bug vanishes. That is exactly why the negative controls never
   caught it — broken clusters have short logs, and their expected layer-8 answer is
   `WARN`/`FAIL` anyway. The bug only appears on a *healthy* cluster running a *real*
   workload: the one case where a wrong answer is most expensive. All six
   `echo "$VAR" | grep -q` sites were converted to here-strings (`<<<`), which are files, not
   pipes.

**The generalisable lesson:** *"no evidence found"* and *"evidence shows failure"* are
different verdicts, and a tool that silently converts the second into the first is worse than
no tool — it sends support hunting for a bug that does not exist. Negative controls alone
would never have caught this. **Validate against a working system too.** Both runs:
[`assets/lab-22/verify_layer8_fix_controls.txt`](../../assets/lab-22/verify_layer8_fix_controls.txt).

### The third bug — and the only one that reported on the *wrong cluster*

Re-running all three controls one last time before committing turned up a defect worse than
either of the above. `CLUSTER=` / `ZONE=` were only ever threaded into the **`gcloud`** calls
(layers 1, 2, 4, 5). Every **`kubectl`** call — the machine-family/tier detection plus layers
3, 6, 6c, 7, 8 — kept reading the **current context**. So asking about one cluster produced a
report spliced from two:

```
$ kubectl config current-context
gke_…_asia-southeast1-c_hypercomputer-a3-tcpxo          # healthy TCPXO

$ CLUSTER=hypercomputer-a3-asiaeast1 ZONE=asia-east1-c scripts/verify_gpu_fabric.sh
-- EXPECTED fabric tier for this family: TCPXO          # WRONG: a3-highgpu-8g is TCPX
 [FAIL] 1. Dataplane V2      '<empty>' (legacy)         # ← asiaeast1, via gcloud
 [PASS] 6c. rxdm devices     …present, 8 aperture devices
                                                        # ← the OTHER cluster, via kubectl
 tier=TCPXO  pass=4  fail=3  warn=1
```

It claimed a device present on a cluster that has none, and mis-detected the tier — so the
NIC target *and* the expected transport pattern were wrong for every check. Note how
**plausible** the output is: sensible FAIL rows, a sensible exit code, nothing inviting
suspicion. Bugs 1 and 2 lost information; this one **mis-attributes one cluster's health to
another cluster's ticket**, which is strictly more expensive.

The fix is not "pass `--context` at each of the 19 call sites" — that is one edit away from
regressing. Resolve the cluster to a context once, then **shadow `kubectl` with a wrapper**
that always pins it, so a later call site cannot opt out. And when no context matches the
requested cluster, **exit 2** rather than quietly falling back to whatever is selected. The
same fix landed in `collect_fabric_bundle.sh`, where the failure mode was worse still: a
support bundle *named* for one cluster containing another cluster's Pods, DaemonSets and logs.

All three controls were then re-run with the current context deliberately left on the TCPXO
cluster, to prove the pin holds: **positive 7 PASS / 0 FAIL exit 0**; **asiaeast1 1 PASS /
5 FAIL exit 1, now correctly `tier=TCPX`**; **asiasoutheast1 1 PASS / 6 FAIL exit 1**. (Layer
8 is `WARN` in all three — no pod was passed, which is the correct *no evidence* verdict, as
distinct from FAIL. The layer-8 `PASS` on live FasTrak traffic is the separate run above.)

> **If a tool takes a target as a parameter, every backend it queries must honour that
> parameter.** Otherwise it will describe the wrong system with total confidence. Pin once at
> a wrapper; fail loudly when the target can't be resolved.

Fix the checker until its negative controls are right, *then* trust its positives.

---

## 5. The decisive layer — and why §4 isn't enough

Layers 1–7 are **necessary**. Only layer 8 — what transport NCCL actually chose — is
**sufficient**. A cluster can pass all of §4 and still fall back to sockets (wrong env
family, missing rxdm sidecar) or, per §3.6, pass layers 1–7 with a fabric that cannot move a
byte. **Never close a ticket on configuration alone.**

The workload must run with:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
```

Without both, the transport choice is *never logged* and the ticket cannot be resolved from
logs at all. Then:

```bash
# --all-containers matters: without it you read the rxdm sidecar, which never logs a
# transport line, and conclude "no evidence" on a healthy fabric (§4).
kubectl logs -n <ns> <pod> --all-containers --tail=4000 \
  | grep -E 'NET/(Socket|IB|FasTrak|GPUDirectTCPX)|GPU Direct RDMA'
```

| Line | Verdict |
|---|---|
| `NET/FasTrak` (TCPXO) / `NET/GPUDirectTCPX` (TCPX) / `NET/IB` | ✅ engaged — slowness is *above* the fabric |
| `NET/Socket` | ❌ silent fallback — this is the bug |
| `GPU Direct RDMA Disabled` | ❌ rxdm sidecar missing from the **workload** Pod |
| nothing | ⚠️ debug env not set, **or the run's output never reached pod stdout** — no evidence exists yet |

The last row has a third cause that cost time here: a job launched via `kubectl exec` writes
NCCL's output to *that session*, so `kubectl logs` is empty even after a perfectly healthy
run. When driving a benchmark by hand, tee into PID 1 so the evidence outlives the shell:

```bash
... torchrun ... 2>&1 | tee /proc/1/fd/1 > /work/run.log
```

> **Flex-start warning.** When the pool scales back to 0 the node takes its NIC list,
> `dmesg` and NCCL logs with it. **Capture while the node is alive.**

Captured live on this cluster — the whole ticket, closed in one line
([`tcpxo_transport_fastrak.txt`](../../assets/lab-22/tcpxo_transport_fastrak.txt)):

```
NCCL INFO Using network FasTrak                     <- both ranks
NCCL INFO Initializing network FasTrak, version: 1.0.17
grep -c 'NET/Socket' -> 0                            <- the number that matters
FASTRAK_IFNAME=eth1,eth2,...,eth8  CTRL_DEV=eth0  SOCKET_IFNAME=eth0   (auto-discovered)
```

That last line is a rule, not a detail: **source the vendor's `nccl-env-profile.sh`; never
hand-write `NCCL_FASTRAK_IFNAME`.** The profile discovers the NIC list on the node it is
running on. A hard-coded list is a fallback waiting for the next machine shape.

> **Scope correction (lab-18, 2026-07-31): this rule is TCPXO-only.** The **TCPX** plugin ships
> **no** `nccl-env-profile.sh` — `ls /usr/local/nvidia/lib64/*env*` returns **0 matches** on a
> working A3 High node — so on that tier hand-writing
> `NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=eth1,eth2,eth3,eth4` is not a shortcut, it is the only
> option (**G28**). The transferable rule is therefore weaker than stated here: *don't invent a
> NIC list — either source a profile that discovers it, or verify your hand-written list against
> the node's `networking.gke.io/nic-info` annotation.* Telling an A3 High customer to "source the
> profile" sends them looking for a file that does not exist.

### 5.1 The measured result — the cliff, closed

With layer 8 green, the benchmark finally means something. 2 × `a3-megagpu-8g`, **16 GPUs**,
the guide's own `allreduce_bench.py` (same harness as [lab-06](../lab-06-2node-nccl-collectives/)
and [lab-12](../lab-12-scaling-sweep/), so the comparison is apples-to-apples):

| Message size | algbw (GB/s) | **busbw (GB/s)** |
|---|---|---|
| 64 MB | 100.61 | 188.65 |
| 128 MB | 120.49 | 225.92 |
| 256 MB | 134.46 | 252.12 |
| 512 MB | 150.75 | 282.65 |
| 1 GB | 163.31 | 306.21 |
| **2 GB** | **169.51** | **317.84** ← peak (12.669 ms) |

| Fabric | 2-node / 16-GPU peak busbw | vs TCPXO |
|---|---|---|
| single-gVNIC TCP (`NET/Socket`, lab-12) | 23.70 GB/s | — |
| **GPUDirect-TCPXO (`NET/FasTrak`, here)** | **317.84 GB/s** | **≈ 13.4×** |

Three things to read off it:

1. **The cliff was an architecture choice, not physics.** Same GPUs, same benchmark, same
   collective — 13.4× is the fabric alone. Every "inter-node is just slow" conclusion earlier
   in this guide was a conclusion about a *configuration*.
2. **Inter-node is now within ~1.5× of intra-node NVLink** (~480 GB/s), instead of ~20×
   below it. That is the difference between "multi-node is a last resort" and "multi-node
   scales".
3. ~~**This is a floor, not a ceiling.**~~ **Corrected 2026-08-02 — it is neither; it is the
   enforced operating point.** The original claim was that the `CPU affinity ... is not a
   subset` warnings meant ranks were un-pinned and a tuned run should beat this. But the TCPXO
   plugin **enforces 14 NCCL variables** and aborts init on any mismatch, so there is no env
   sweep to run ([G34](../../reference/lab-build-gotchas.md), evidence in `assets/lab-23/`).
   The one knob outside the policy file is `NCCL_ALGO`, worth **+8.2%** at 24 GPUs. Quote
   317.84 GB/s as this shape's measured number — not as a maximum to beat, and not as a
   pessimistic floor either. See [lab-23](../lab-23-enabled-scaling-curve/).

Full table with provenance (plugin/rxdm/driver/GKE versions, MTU, transport):
[`tcpxo_allreduce_16gpu.txt`](../../assets/lab-22/tcpxo_allreduce_16gpu.txt). In-pod
interfaces, MTU 8244 on `eth1–eth8`, the 8 injected aperture devices and per-rail balance:
[`tcpxo_inpod_fabric.txt`](../../assets/lab-22/tcpxo_inpod_fabric.txt).

> **Read the log's error count sceptically.** The rank-0 log contains 63 lines matching
> `-i error|abort`. **None is a fault:** 47 `abort` + 16 `Abort` are normal comm teardown
> (`Abort START`), and the `NCCL WARN` lines are the config-checker shim's advisories
> (`NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY ... expected unset`, `NCCL_LIB_DIR`,
> `TORCH_NCCL_ASYNC_ERROR_HANDLING`, CPU affinity). Grepping for "error" and escalating the
> count is a classic false escalation — quote the *transport* line instead.

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

An unrun alert rule is a guess wearing a uniform. This lab's original §7 said: alert on
**GPUs busy, GPU NICs idle**. Then we ran it against live Managed Prometheus during the
§5.1 all-reduce — and found the rule was not miscalibrated, it was **inverted**.

### 7.1 The measurement that killed the obvious alert

100% GPU util, ~283 iterations/s of a 2 GB all-reduce, 317 GB/s busbw. The eight GPU NICs
during that:

| Where you look | eth1–eth8 TX, under full FasTrak load |
|---|---|
| in-pod `/sys/class/net/ethN/statistics/tx_bytes` | **0.00 Gbit/s — 4 packets/s** |
| GMP `rate(container_network_transmit_bytes_total[5m])` | **0.000** |
| hypervisor `instance_network_sent_bytes_count` | 0.001 Gbit/s |
| host netns | `eth1-8` **do not exist** — passed into the Pod |

Four packets per second on eight rails while ~1.1 TB/s crosses them. **TCPXO/FasTrak is a
kernel-bypass datapath** — rxdm drives the NIC from userspace and DMAs into GPU HBM via
dmabuf, so payload bytes never traverse the kernel stack that fills `netdev` counters. The
residue is ARP and link housekeeping.

So "GPUs busy **and** GPU-NIC bytes near zero" is the signature of a **healthy** TCPXO
fabric. The original rule would have paged continuously on a working cluster and said
nothing during the 13.4× regression. Delete it; don't retune it.

> **Scope correction (lab-18, 2026-07-31) — this is a FasTrak property, not a GPUDirect one.**
> The same probe on a live **TCPX** fabric read **~50.4 GB per rail** on `eth1–eth4`, balanced
> to within **0.05%**, while `eth0` carried 0.04 GB. TCPX still moves its payload over host TCP
> sockets (GPUDirect accelerates the *memory* path via dmabuf, not the transport), so the kernel
> netdev counters see everything. Consequence: **"GPU-NIC bytes near zero" means healthy on A3
> Mega and BROKEN on A3 High** — the same query, opposite verdicts, per tier. Evidence:
> [`assets/lab-18/after_tcpx_monitoring.txt`](../../assets/lab-18/after_tcpx_monitoring.txt).

That also retires the rail-balance idea *for this tier*: `max/avg` over eight rails that
all read zero is 0/0. It stays valid on A3 Ultra/A4 RoCE **and on TCPX — where it is now
measured, not assumed** — so doc-25 keeps it with a per-tier applicability note. The healthy-balance evidence from this lab is
still good, it just comes from the **NCCL ring setup**, not from a counter: all eight rails
`eth1`–`eth8` referenced **43 times each** — dead even.

### 7.2 What does work: DCGM PCIe, proven against both states

The bytes must cross PCIe to reach the NIC, and DCGM counts PCIe. Same job, same Pods, same
GPUs, **only environment variables changed** (`NCCL_NET_PLUGIN=none NCCL_SOCKET_IFNAME=eth0`):

| Signal (per node, 8 GPUs) | FasTrak | Socket path | Ratio |
|---|---|---|---|
| `DCGM_FI_PROF_PCIE_TX_BYTES` | **1327.3 Gbit/s** | **29.7 Gbit/s** | **44.7×** |
| `DCGM_FI_DEV_GPU_UTIL` | 100 % | 100 % | **1.0×** |
| Pod `eth0` TX (in-pod) | ~0.00 Gbit/s | 23.90 Gbit/s | — |
| Pod `eth1-8` TX (in-pod) | 0.00 | 0.00 | — |
| iterations in 449 s | ~127,000 | 2,400 | ~53× |

GPU util is **identical on both sides of a 53× throughput cliff** — that column is the whole
reason this failure is silent. PCIe TX separates the states by 44×:

```
GPU util high + PCIe TX high + eth1-8 ~0   =>  HEALTHY GPUDirect
GPU util high + PCIe TX low  + eth0 high   =>  SILENT FALLBACK
```

The corrected rules, with thresholds derived from those two numbers, plus the Grafana row,
are in
[doc-25 §5](../../docs/part5-operations-diagnostics/25-fabric-diagnostics-playbook.md#5-monitoring--catching-silent-fallback-before-a-human-notices).
The three that matter:

- `GPUFabricSilentFallbackSuspected` — `DCGM_FI_DEV_GPU_UTIL > 80` **and**
  `sum by (Hostname) (DCGM_FI_PROF_PCIE_TX_BYTES) < 200e9` for 15m. Threshold sits ~6.6×
  below healthy and ~6.7× above fallback. The only automated warning you will ever get for
  the ~13× regression measured in §5.1.
- `GPUControlNICCarryingBulkTraffic` — `eth0` above 5 Gbit/s. Independent corroboration:
  measured 23.9 Gbit/s during fallback vs ~0 when healthy.
- `GPUNCCLPluginNotReady` — plugin DaemonSet ready count 0 **while GPU nodes exist**. The
  second clause is what keeps stocked-out Flex pools from paging anyone (§3.5) — but it used
  `kube_node_labels`, which has **0 series** here, so the guard never evaluated. Now keyed on
  `count(DCGM_FI_DEV_GPU_UTIL) > 0`.

### 7.3 Three more metric names that silently don't exist

`node_network_*`, `kube_node_labels`, `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` and
`DCGM_FI_DEV_XID_ERRORS` all return **0 series** on GKE managed collection. A PromQL rule
over a non-existent metric doesn't error — it just never fires. Count series before you
trust a rule.

And when you do use `container_network_*`, aggregate with **`max by (node, interface)`,
never `sum`**: cAdvisor reports the Pod's netns counters once per container — 16 series per
node here — which turned a real 13.86 Gbit/s into **235.57 Gbit/s**. That inflation makes
any "NIC traffic is low" threshold ~10× too lax.

Every number above, the cross-cluster control, and the four validated queries:
[`assets/lab-22/tcpxo_monitoring_validation.txt`](../../assets/lab-22/tcpxo_monitoring_validation.txt).

> **The transferable lesson.** Kernel counters are blind to kernel-bypass datapaths —
> TCPXO, RDMA, DPDK, SPDK — but **not** to every accelerated one: TCPX keeps the kernel TCP
> transport and stays fully visible, which is why this lesson is about the *datapath*, never
> about the marketing tier. Before alerting on the *absence* of a signal, prove your collector
> can physically observe its *presence* **on the tier you are alerting on**. And validate every threshold against
> **both** states of the thing you're detecting; forcing the same job onto the slow path
> costs one environment variable and no cluster changes.

---

## 8. What support should take from this lab

1. **"No errors" is not "no problem."** The fabric's failure mode is a number, not a log line.
   §5.1 puts a size on it: **13.4×**.
2. **"Job succeeded" proves nothing** about the fabric. Nor does a single-node benchmark —
   NVLink is unaffected by fabric bugs.
3. **Exit `0` proves nothing either** (§3.6). The rxdm sidecar reports `Exiting with
   result:1` in its logs and exits `0` to the kubelet. Read container *logs*, not just
   container *state*.
4. **`Ready 1/2` with no error state is a fabric symptom**, not a scheduling hiccup. It is
   the single highest-yield thing to look for on a TCPXO Pod.
5. **Layers 1, 2 and 4 are create-time-only.** If they fail, say "new cluster / recreate the
   pool" on day one rather than debugging layers 5–8 for a week.
6. **Only the NCCL transport line closes a ticket.** Everything else is circumstantial.
7. **Never hand-roll the installer, and pin rxdm/plugin to the node's driver** (§3.1, §3.6) —
   both are silent, fail-open mistakes.
8. **Distinguish stocked-out from misconfigured** — a Flex pool at 0 nodes is not a bug.
9. **Read which container failed**, not just which Pod (§3.2).
10. **Validate your tools against known-broken systems** before trusting them on unknown ones
    (§4) — and never let a tool assert a fault it did not observe.
11. **Validate them against a known-*working* system too.** Both negative controls passed
    while layer 8 — the decisive check — was silently broken by a `grep -q` pipeline (§4).
    "No evidence" is a verdict too, and a wrong one is expensive.
12. **Don't ask a customer for GPU-NIC throughput graphs.** On TCPXO they read zero when the
    fabric is *healthy* (§7.1). Ask for `DCGM_FI_PROF_PCIE_TX_BYTES` and the NCCL transport
    line instead — and if a ticket says "our GPU NIC dashboards show no traffic", that is not
    the bug.
13. **An alert rule is a claim about the system.** Ship it unrun and it is a guess: this lab's
    own rule was *inverted*, and three of the metric names it referenced return 0 series on
    managed collection (§7.3). Fire the rule against a real fault before you trust it.
