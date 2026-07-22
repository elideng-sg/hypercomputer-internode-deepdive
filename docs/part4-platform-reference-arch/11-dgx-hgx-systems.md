# 11: DGX & HGX Systems — System-Level Architecture and Troubleshooting

**⚠️ Knowledge-first / reference architecture.** DGX system software (DGX OS, NVSM, GPU Fabric Manager) is not present on this GCP A3 cluster. This document teaches the DGX/HGX platform architecture and troubleshooting methodology as reference knowledge. Hands-on verification of what IS observable on our A3 cluster — `nvidia-smi topo`, DCGM, NVLink counters — is covered in lab-11. This doc never claims DGX hardware or DGX system software was run here.

## Introduction

When people say "DGX," they often mean one of three things: the **HGX GPU baseboard** (the physical 8-GPU + NVSwitch assembly), a complete **DGX H100/H200 system** (the full server chassis with dual CPUs, storage, NICs, and DGX-specific firmware/software), or the **DGX OS** software stack that manages the system. Understanding these distinctions is critical for troubleshooting GPU infrastructure and for correctly mapping capabilities between cloud (GCP A3/A4) and on-premises NVIDIA platforms.

This document walks through:

1. **HGX baseboard** vs full **DGX system** hardware architecture
2. **DGX OS** and the system-level management stack: **NVSM** and **GPU Fabric Manager**
3. **System-level troubleshooting** methodology: health checks, NVLink/NVSwitch diagnostics, BMC sideband access, and common failure modes
4. **Mapping to GCP**: which of these components Google manages/hides for A3 tenants, and what tenants CAN observe

Part I (section 04, when written) covered the HGX baseboard's NVLink/NVSwitch fabric at the **intra-node** level (GPU-to-GPU communication within a single node). This document elevates to **system-level** concerns: firmware, BIOS, power/thermal management, fabric health monitoring, and the orchestration software that ties a DGX node (or an entire DGX SuperPOD) into a production AI factory.

---

## 1. HGX Baseboard vs Full DGX System

### 1.1 HGX Baseboard: The 8-GPU Foundation

The **HGX** (High-Performance GPU eXtension) baseboard is NVIDIA's modular GPU subsystem, shipped to OEMs and cloud providers as a standardized building block. It contains:

- **8 × datacenter GPUs** (H100, H200, or B200, depending on generation)
- **NVSwitch interconnect fabric** (4th-generation NVSwitch on HGX H100/H200; 5th-gen on HGX B200)
  - Each GPU connects to **all NVSwitches** via NVLink; NVSwitches provide full-bisection-bandwidth, non-blocking connectivity between all 8 GPUs
  - **HGX H100**: 4th-gen NVLink (50 GB/s per link × 18 links per GPU = 900 GB/s bidirectional per GPU); ~900 GB/s aggregate bisection bandwidth
  - **HGX H200**: same fabric, upgraded GPU memory (H200 = H100 die + 141 GB HBM3e vs 80 GB HBM3)
  - **HGX B200**: 5th-gen NVLink, higher bisection bandwidth (~1.8 TB/s per GPU)
- **PCIe connectivity to host CPUs** (each GPU exposes a PCIe Gen5 x16 interface)
- **Power delivery and cooling mounts** (HGX is a **mezzanine card** that slots into a server chassis; power/cooling provided by the host system)

The HGX baseboard is **not a standalone server**. It requires a host system with:
- Dual-socket CPUs (Intel Xeon Sapphire Rapids or AMD EPYC Genoa on DGX H100/H200)
- PCIe risers to connect HGX GPUs to CPU root complex
- Power supplies (7-10 kW for an 8-GPU HGX H100 system)
- Liquid or air cooling (DGX H100 uses direct liquid cooling for GPUs)

**What our GCP A3 node physically is:** Each `a3-highgpu-8g` node runs an **HGX H100 baseboard** (8×H100 80GB + 4th-gen NVSwitch). Google integrates the HGX baseboard into their server chassis with host CPUs, storage, and NICs (gVNIC on A3 High/Mega; Mellanox ConnectX-7 on A3 Ultra/A4). The HGX hardware is identical to what NVIDIA ships in DGX H100, but the **system software layer** (DGX OS, NVSM, Fabric Manager) is **not exposed** to GCP tenants — Google manages firmware, fabric health, and NVSwitch configuration at the hypervisor/control-plane level.

### 1.2 Full DGX H100/H200 System

A **DGX H100** or **DGX H200** system is NVIDIA's complete, turnkey AI server. It bundles the HGX baseboard with:

**Hardware:**
- **Dual Intel Xeon Platinum 8480C (Sapphire Rapids) CPUs** — 56 cores each, 112 cores total, 2 TB system RAM
- **HGX H100 or HGX H200 baseboard** (8 GPUs + NVSwitch fabric, as above)
- **8 × Mellanox ConnectX-7 NICs** (400 Gbps each, dual-port; total 3.2 Tbps network fabric connectivity)
  - 4 InfiniBand HDR/NDR ports (for DGX SuperPOD rail-optimized fabrics) OR 4 Ethernet RoCE ports (for Spectrum-X fabrics)
  - 4 additional Ethernet management ports
- **2 × BlueField-3 SuperNICs** (DPU-accelerated NICs; offload networking, storage, security; optional on some SKUs)
- **30 TB NVMe storage** (U.2 drives for local dataset caching, model checkpoints)
- **10 kW power supplies** (dual redundant PSUs)
- **Liquid cooling** (direct-to-chip liquid cooling for GPUs; air cooling for CPUs/NICs)
- **BMC (Baseboard Management Controller)** — out-of-band management via IPMI/Redfish for power cycling, BIOS config, firmware updates, thermal/power monitoring

**Software (DGX OS stack):**
- **DGX OS** (Ubuntu-based, NVIDIA-customized; pre-installs drivers, CUDA, Docker, NCCL, cuDNN, TensorRT, and DGX management tools)
- **NVSM** (NVIDIA System Management Interface) — GUI + CLI for system health, firmware, BIOS, fan/power/thermal monitoring
- **GPU Fabric Manager** (`nv-fabricmanager`) — user-space daemon managing NVLink/NVSwitch topology, health, error handling, and link training
- **Base Command Manager agent** (optional; connects DGX node to BCM cluster orchestrator for fleet-scale job scheduling, monitoring, and firmware updates)

**Key distinction:** A DGX system is **fully integrated and validated** by NVIDIA — all firmware (GPU, NVSwitch, BMC, NIC) is tested together, and NVSM provides a single pane of glass for health monitoring. An OEM server with an HGX baseboard **can match DGX performance** if properly configured, but the OEM (or cloud provider) must handle firmware integration, fabric tuning, and health monitoring themselves. GCP A3 nodes use HGX baseboards in Google-designed servers with Google-managed firmware; DGX uses NVIDIA-designed servers with NVIDIA-managed firmware.

---

## 2. DGX System Software Stack

### 2.1 DGX OS

**DGX OS** is NVIDIA's Ubuntu-based Linux distribution for DGX systems. It is **not** a general-purpose OS — it is purpose-built for AI workloads and ships with:

- **Pre-installed drivers** (NVIDIA Data Center GPU Driver, version matched to DGX firmware)
- **CUDA Toolkit** (latest production release)
- **Deep learning frameworks** (via NGC containers: PyTorch, TensorFlow, JAX, NeMo)
- **NCCL** (with DGX-specific tuning and topology files for SuperPOD fabrics)
- **Docker + NVIDIA Container Toolkit** (`nvidia-docker`, device passthrough for GPUs in containers)
- **DGX system management tools** (NVSM, Fabric Manager, BMC CLI tools)

**Updates are managed via NVIDIA Update Manager**, which pulls validated firmware + driver + CUDA stacks from NVIDIA's repos. Unlike a vanilla Ubuntu system, where a user might `apt upgrade` drivers independently and break CUDA compatibility, DGX OS ensures the entire stack (firmware, driver, CUDA) is updated atomically.

**Contrast with GCP A3:** GCP A3 nodes run a Google-managed OS image (Container-Optimized OS or Ubuntu with GKE-managed drivers). Drivers are installed via **GKE device-plugin DaemonSets**, not via DGX OS. Tenants do not see NVSM or Fabric Manager (Google manages those at the hypervisor layer, if present at all).

### 2.2 NVSM (NVIDIA System Management Interface)

**NVSM** is the system-level health and management tool for DGX. It provides:

- **Health checks** (automated daily health validation; on-demand diagnostics via GUI or `nvsm show health`)
- **Firmware inventory** (GPU, NVSwitch, BMC, NIC firmware versions)
- **Power/thermal monitoring** (GPU power draw, CPU socket temps, fan speeds, liquid cooling flow rates)
- **NVLink/NVSwitch fabric health** (link training status, error counters, degraded fabric detection)
- **BIOS/BMC configuration** (boot order, power limits, fan curves, out-of-band management settings)
- **Field diagnostics** (integrated with `nvidia-smi` and `dcgmi diag` for GPU-level tests; adds system-level checks like NVSwitch stress tests, memory bandwidth validation across all GPUs)

**Example NVSM CLI usage (illustrative — not captured on this cluster):**

```bash
# Show system health summary
$ nvsm show health
System Health: OK
  GPUs: 8/8 healthy
  NVLink: All links trained, no errors
  NVSwitch: 4/4 switches operational
  BMC: reachable, no critical alerts
  Fans: all nominal (35-60% speed)
  Power: 8.2 kW (82% of rated capacity)

# Show NVLink topology (similar to nvidia-smi topo -m, but adds NVSwitch details)
$ nvsm show nvlink
GPU 0 <-> NVSwitch 0,1,2,3 (18 links total, all UP)
GPU 1 <-> NVSwitch 0,1,2,3 (18 links total, all UP)
...
NVSwitch 0: 144 ports, 144 active
NVSwitch 1: 144 ports, 144 active
...

# Show firmware versions
$ nvsm show firmware
Component           | Version
--------------------|------------------
GPU 0 InfoROM       | H100.0200.00.03
NVSwitch 0 FW       | 1.0.15
BMC FW              | 1.23.45
ConnectX-7 NIC 0 FW | 28.35.1000
```

**Why tenants don't see NVSM on GCP:** NVSM talks to the BMC and requires privileged access to NVSwitch firmware and PCIe config space. Exposing NVSM to tenants would allow one tenant to observe (or even modify) hardware state that affects other tenants on the same physical host. GCP abstracts this via the hypervisor and control plane.

### 2.3 GPU Fabric Manager

**GPU Fabric Manager** (`nv-fabricmanager`) is a **critical user-space daemon** that manages the NVLink/NVSwitch fabric on multi-GPU systems. Its responsibilities include:

1. **NVLink training and topology discovery** at boot time (GPUs and NVSwitches negotiate link speeds, widths, and routing tables)
2. **Error handling and link recovery** (if an NVLink goes down due to a transient error, Fabric Manager attempts to retrain the link; if retraining fails, it marks the link degraded and reconfigures routing)
3. **Fabric-wide health monitoring** (aggregates error counters from all GPUs and NVSwitches; logs errors to `/var/log/fabricmanager.log`)
4. **Coordination with NVSM** (Fabric Manager reports fabric state to NVSM for health dashboards)

**When is Fabric Manager required?** Fabric Manager is **mandatory on systems with NVSwitch** (i.e., all HGX-based systems: DGX, HGX servers, GCP A3/A4). Without Fabric Manager, NVLink will not train, and `nvidia-smi topo -m` will show all GPU-GPU links as disconnected. On systems with **direct GPU-to-GPU NVLink** (e.g., 2-GPU or 4-GPU servers without NVSwitch), Fabric Manager is optional (the driver can manage those links directly).

**Fabric Manager logs and failure modes:**

Fabric Manager logs to `/var/log/fabricmanager.log` (on DGX systems). Common failure modes include:

- **Failed NVLink training at boot** → Fabric Manager retries link training up to N times; if all retries fail, it marks the link DOWN and logs `NVLINK training failed on GPU X port Y`
  - **Root cause:** bad cable (NVLink uses high-speed signaling; even small signal integrity issues cause training failures), bad NVSwitch port, GPU firmware mismatch
  - **Diagnosis:** check `nvidia-smi nvlink -s` (link status), reseat cables, update GPU/NVSwitch firmware via NVSM
- **Degraded fabric (partial link loss)** → e.g., 17/18 NVLinks trained on a GPU, one link DOWN
  - Fabric Manager reconfigures routing to avoid the dead link; multi-GPU jobs will run but at reduced bandwidth
  - **Diagnosis:** `nvidia-smi nvlink --status` shows link-by-link state; correlate with Fabric Manager logs; if persistent, replace cable or RMA GPU
- **NVSwitch SXID errors** → NVSwitch reports unrecoverable errors (e.g., internal parity error, routing table corruption)
  - Fabric Manager logs SXID (Switch XID) errors; system may require reboot or NVSwitch firmware downgrade
  - **Diagnosis:** parse `/var/log/fabricmanager.log` for SXID codes; cross-reference with NVIDIA support docs; on DGX, NVSM shows switch health

**Example Fabric Manager log snippet (illustrative):**

```
[INFO] GPU 0 NVLink 0-17: training started
[INFO] GPU 0 NVLink 0-17: training complete, all links UP
[INFO] NVSwitch 0: 144 ports active
[WARN] GPU 3 NVLink 12: CRC errors detected, retraining
[INFO] GPU 3 NVLink 12: retraining successful
[ERROR] GPU 5 NVLink 7: training failed after 3 retries, link marked DOWN
[CRITICAL] NVSwitch 2: SXID 0x5C (routing table parity error), requesting system reboot
```

**GCP A3 and Fabric Manager:** Fabric Manager is **running on A3 nodes** (because HGX requires it), but it runs **inside Google's control plane / hypervisor**, not in the tenant VM. Tenants cannot access `/var/log/fabricmanager.log` or restart the `nv-fabricmanager` service. If an NVLink fails, Google's monitoring (analogous to NVSM) detects it and either repairs/reboots the node or marks it for maintenance. Tenants see the effect via `nvidia-smi topo -m` (a missing link) or degraded NCCL performance, but cannot troubleshoot Fabric Manager directly.

---

## 3. System-Level Troubleshooting Walkthrough

This section presents a **reference troubleshooting methodology** for DGX systems, describing what a DGX admin or on-prem GPU cluster operator would do when a node shows degraded performance, health check failures, or intermittent training crashes. On GCP A3, tenants perform a **subset** of these steps (those that don't require BMC or Fabric Manager access).

### 3.1 Daily Health Validation Workflow (DGX)

**Goal:** Verify GPU, NVLink, NVSwitch, NIC, and thermal health before running production workloads.

**Steps:**

1. **Run NVSM health check**
   ```bash
   nvsm show health
   ```
   Expected: `System Health: OK`. If not OK, note which subsystem is unhealthy (GPU, NVLink, thermal, BMC).

2. **Check GPU status and ECC errors**
   ```bash
   nvidia-smi -q -d TEMPERATURE,POWER,ECC,CLOCK
   ```
   - **Temperature:** GPUs should be <80°C idle, <85°C under load (DGX H100 direct liquid cooling targets 60-75°C under load)
   - **Power:** Idle ~50-100 W per GPU; load up to 700 W (H100 TDP). If a GPU is at 700 W idle, suspect a runaway process or stuck kernel.
   - **ECC errors:** `Volatile ECC errors: 0`, `Aggregate ECC errors: <10` over system lifetime is normal. >100 volatile ECC errors in a day suggests bad HBM; >1000 aggregate over months may warrant RMA.
   - **Clock throttle reasons:** Should be `Clocks Throttle Reasons: Not Active`. If throttling due to `HW Thermal Slowdown` or `HW Power Brake Slowdown`, check cooling (fans, liquid flow) or power supply capacity.

3. **Check NVLink health**
   ```bash
   nvidia-smi nvlink --status
   ```
   Expected: all links `Active` (for HGX H100, each GPU should show 18 NVLink connections, all UP). If any link is `Inactive` or `Down`, check Fabric Manager logs.

4. **Check NVSwitch health (via NVSM or nvidia-smi)**
   ```bash
   nvidia-smi nvlink -s  # shows NVSwitch enumeration on systems where exposed
   ```
   On DGX, NVSM also shows per-switch health. All 4 NVSwitches (on HGX H100) should be operational.

5. **Run DCGM diagnostic (level 3)**
   ```bash
   dcgmi diag -r 3
   ```
   This runs GPU stress tests, memory bandwidth checks, and NVLink peer-to-peer transfers. Expected: `PASS` for all GPUs. Failures indicate hardware issues (failing HBM, bad NVLink cable, overheating).

6. **Check NIC status (for InfiniBand or RoCE fabrics)**
   ```bash
   ibstat  # if InfiniBand
   # or
   ibv_devinfo  # shows RDMA devices
   ```
   Expected: all IB ports `ACTIVE`, link speed 400 Gb/s (HDR) or 800 Gb/s (NDR). For RoCE, use `ethtool` to check link status.

**Frequency:** Run this workflow daily (automated via cron or Base Command Manager), or before starting a multi-day training job.

### 3.2 Diagnosing Degraded Fabric (Partial NVLink Loss)

**Symptom:** Training job runs slower than expected; `nvidia-smi topo -m` shows one or more GPU-GPU links missing; `nvidia-smi nvlink --status` shows some links DOWN.

**Root cause candidates:**
- Bad NVLink cable (most common)
- NVSwitch port failure
- GPU NVLink PHY failure (rare; usually RMA required)
- Firmware mismatch (GPU vs NVSwitch firmware incompatibility)

**Diagnosis steps (DGX admin):**

1. **Identify which link is down**
   ```bash
   nvidia-smi nvlink --status | grep -i inactive
   # Example output (illustrative):
   # GPU 3, Link 12: Inactive
   ```

2. **Check Fabric Manager logs**
   ```bash
   grep -i "Link 12" /var/log/fabricmanager.log | tail -20
   ```
   Look for `training failed`, `CRC errors`, `signal integrity` warnings.

3. **Check error counters**
   ```bash
   nvidia-smi nvlink --error-counters
   ```
   High CRC error counts on a specific link suggest cable or signal integrity issue.

4. **Reseat the NVLink cable** (if accessible; on HGX H100, NVLink cables are internal PCB traces, not external cables; "reseating" means power-cycling the node to retrain links).

5. **Update GPU and NVSwitch firmware**
   ```bash
   nvsm update firmware
   ```
   NVSM pulls latest validated firmware from NVIDIA repos and applies updates.

6. **Reboot and retrain fabric**
   ```bash
   reboot
   # After reboot, check nvidia-smi nvlink --status again
   ```

7. **If link still DOWN, mark node for maintenance** and run jobs on other nodes (or run at reduced performance if job can tolerate partial fabric).

**On GCP A3:** Tenant cannot access Fabric Manager logs or update firmware. If a link is down, the tenant sees degraded `nccl-tests` performance. The tenant should file a GCP support ticket; Google will investigate (via their NVSM-equivalent monitoring) and either repair/reboot the node or replace it.

### 3.3 Diagnosing Thermal Throttling

**Symptom:** GPU clocks drop during training; `nvidia-smi` shows `Clocks Throttle Reasons: HW Thermal Slowdown`.

**Root cause candidates:**
- Insufficient cooling (fan failure, blocked air intake, liquid cooling pump failure)
- Ambient temperature too high (data center HVAC failure)
- TDP limit set too low (misconfigured power cap)

**Diagnosis steps (DGX admin):**

1. **Check GPU temperature history**
   ```bash
   nvidia-smi dmon -s pucvmet -c 10  # monitor power, util, clocks, violations, memory, temp
   ```
   If temperature >85°C and rising, suspect cooling issue.

2. **Check cooling subsystem via NVSM**
   ```bash
   nvsm show cooling
   # Illustrative output:
   # Fan 0: 4500 RPM (60% speed)
   # Fan 1: 0 RPM (FAILED)
   # Liquid cooling: flow rate 2.5 L/min (nominal 2.0-3.0)
   ```
   Failed fan or low liquid flow indicates hardware issue.

3. **Check BMC thermal sensors**
   ```bash
   ipmitool sdr type temperature
   # Shows all thermal sensors (CPU, GPU inlet, exhaust, PCIe riser temps)
   ```
   If exhaust temp >60°C, check data center HVAC.

4. **If cooling hardware is healthy, increase fan speed manually** (temporary workaround)
   ```bash
   nvsm set fan-speed --all 80  # set all fans to 80% (louder, but better cooling)
   ```

5. **If still throttling, reduce GPU power cap** (trades performance for stability)
   ```bash
   nvidia-smi -pl 650  # set power limit to 650W (down from 700W)
   ```

6. **File RMA for failed cooling component** (fan, pump, etc.).

**On GCP A3:** Tenant cannot access BMC or set fan speeds. If thermal throttling occurs, file a support ticket. Google monitors thermal sensors at the hypervisor/control-plane level and replaces failing hardware.

### 3.4 Using BMC / IPMI for Out-of-Band Troubleshooting

**BMC (Baseboard Management Controller)** is an independent ARM-based controller on the DGX motherboard that provides **out-of-band management** — it remains powered and accessible even if the OS crashes or the CPUs are offline. BMC exposes an **IPMI (Intelligent Platform Management Interface)** or **Redfish API** for remote management.

**Common BMC tasks:**

1. **Remote power control**
   ```bash
   ipmitool -H <dgx-bmc-ip> -U admin -P <password> power status
   ipmitool -H <dgx-bmc-ip> -U admin -P <password> power cycle
   ```

2. **Serial console access** (view POST/BIOS output, boot logs, kernel panics)
   ```bash
   ipmitool -H <dgx-bmc-ip> -U admin -P <password> sol activate
   ```

3. **Sensor monitoring** (thermal, power, voltage)
   ```bash
   ipmitool -H <dgx-bmc-ip> -U admin -P <password> sdr list
   ipmitool -H <dgx-bmc-ip> -U admin -P <password> sensor reading "GPU 0 Temp"
   ```

4. **Firmware updates** (BMC firmware, BIOS updates via BMC web UI or CLI)

**Security note:** BMC access is highly privileged (power-cycling a node kills all running jobs). On DGX clusters, BMC access is restricted to cluster admins and isolated on a separate management network (out of band from the compute/storage/IB fabrics).

**On GCP A3:** Tenant has no BMC access. Google's control plane performs equivalent functions (power-cycle unresponsive nodes, monitor sensor telemetry, update firmware).

---

## 4. Mapping DGX Capabilities to GCP A3

The table below summarizes which DGX system-level capabilities are **exposed to GCP A3 tenants**, **abstracted by Google**, or **not applicable**.

| Capability / Tool | DGX H100/H200 (On-Prem) | GCP A3 (a3-highgpu-8g) | Notes |
|:---|:---|:---|:---|
| **HGX H100 Baseboard** | Present (8×H100 + NVSwitch) | Present (8×H100 + NVSwitch) | Identical hardware; same NVLink/NVSwitch topology |
| **DGX OS** | Full access (Ubuntu-based, pre-installed drivers/CUDA) | Not present (GKE-managed Container-Optimized OS or Ubuntu + GKE device-plugin DaemonSets) | Tenant manages OS in containers/VMs, not DGX OS stack |
| **NVSM (System Management)** | Full access (`nvsm show health`, firmware updates, thermal monitoring) | Not exposed (Google manages via control plane) | Tenant cannot run `nvsm` commands; Google monitors equivalent metrics |
| **GPU Fabric Manager** | Admin access (`nv-fabricmanager` service, logs in `/var/log/fabricmanager.log`) | Running but not exposed (Google-managed, likely in hypervisor) | Tenant sees effects (NVLink topology via `nvidia-smi topo`) but cannot access Fabric Manager logs or restart service |
| **BMC / IPMI** | Admin access (out-of-band power control, sensor monitoring, serial console) | Not exposed (Google-managed) | Tenant cannot power-cycle nodes or access BMC; Google control plane performs equivalent functions |
| **nvidia-smi (GPU Monitoring)** | Full access | Full access | Identical tool; tenant sees same GPU metrics (temp, power, ECC, clocks, NVLink status) |
| **DCGM Diagnostics** | Full access (`dcgmi diag -r 3`) | Full access (via DCGM DaemonSet in GKE) | Tenant can run level-3 diagnostics; same hardware tests as DGX |
| **NVLink Topology** | Full access (`nvidia-smi topo -m`, `nvidia-smi nvlink --status`) | Full access | Tenant sees same 8-GPU NVLink mesh; can query link status and error counters |
| **NVSwitch Visibility** | Full (NVSM shows per-switch health; Fabric Manager logs switch errors) | Partial (tenant sees GPU-NVSwitch connectivity via `nvidia-smi topo`, but no per-switch health metrics) | Tenant cannot enumerate NVSwitches or query switch firmware/errors directly |
| **Firmware Updates** | Admin via NVSM or `nvidia-firmware-update` | Managed by Google (control plane) | Tenant cannot update GPU, NVSwitch, or NIC firmware; Google rolls updates via GKE releases |
| **NIC Management** | Admin access (ConnectX-7 firmware, `mlxconfig`, `ibstat` for IB) | Partial (ConnectX-7 on A3 Ultra/A4; gVNIC on A3 High/Mega; no firmware access) | Tenant can query NIC via `ethtool` or `ibv_devinfo` (if RoCE), but cannot update firmware or change NIC modes (IB vs Ethernet) |
| **Cooling / Thermal Management** | Admin access (NVSM fan control, liquid cooling monitoring, BMC thermal sensors) | Not exposed (Google manages cooling; tenant sees throttle warnings via `nvidia-smi` if they occur) | Tenant cannot set fan speeds or query cooling subsystem health |
| **Field Diagnostics (NVSM)** | Full (NVSM integrates DCGM + system-level checks like NVSwitch stress tests) | Partial (tenant runs DCGM diagnostics; no system-level checks) | DCGM-level tests (GPU stress, memory BW, peer-to-peer) identical; NVSM-specific tests (NVSwitch stress, BMC integration) not available |

**Key takeaway:** The **GPU hardware and DCGM/nvidia-smi software layer** are functionally identical between DGX and GCP A3. The **system management layer** (NVSM, Fabric Manager, BMC) is abstracted away on GCP — Google handles firmware, fabric health, and power/thermal management at the control-plane level. Tenants retain **full observability** of GPU health, NVLink topology, and DCGM diagnostics, but lose **direct control** of firmware updates, NVSwitch troubleshooting, and cooling configuration.

---

## 5. Base Command Manager (BCM) — Fleet Orchestration

**Base Command Manager (BCM)** is NVIDIA's cluster-scale orchestration and management platform for DGX systems. It provides:

- **Job scheduling and queueing** (gang scheduling, multi-node job submission, Slurm or Kubernetes integration)
- **Fleet-scale monitoring** (aggregates NVSM health, DCGM metrics, and fabric status across 10s to 1000s of DGX nodes)
- **Centralized firmware and driver updates** (push validated firmware/driver stacks to entire fleet; rollback if issues detected)
- **User management and quotas** (multi-tenant resource allocation, per-user/per-project GPU quotas)
- **Data management** (shared storage integration, dataset staging, checkpoint backup)

BCM is deployed as a **separate management cluster** (typically Kubernetes-based, running on non-GPU nodes) with **BCM agents** installed on each DGX node. Agents report health metrics to BCM and execute firmware updates, job launches, and diagnostic commands issued by the BCM control plane.

**BCM vs GCP orchestration:**

| Function | NVIDIA Base Command Manager (DGX) | GCP AI Hypercomputer (A3/A4) |
|:---|:---|:---|
| **Job Scheduling** | BCM job scheduler (Slurm integration or native K8s) | GKE + JobSet + Kueue (gang scheduling) |
| **GPU Provisioning** | Static capacity (DGX nodes always powered on; BCM allocates GPUs from pool) | Dynamic Workload Scheduler (DWS) — auto-provisions GPU nodes from quota, 7-day holds |
| **Fleet Monitoring** | BCM dashboard (NVSM + DCGM metrics aggregated from all nodes) | GKE Monitoring + Cloud Logging + dcgm-exporter DaemonSet + Prometheus/Grafana |
| **Firmware/Driver Updates** | BCM pushes updates fleet-wide (admin-controlled cadence) | Google control plane pushes updates (tied to GKE release schedule) |
| **Multi-Tenancy** | BCM user quotas + Slurm accounts / K8s namespaces | GKE namespaces + Kueue quotas + IAM (per-project GPU quotas) |

Both platforms achieve the same goals (orchestrated job execution, fleet health monitoring, firmware management), but BCM is **on-prem, admin-controlled** while GCP is **cloud-managed, tenant-controlled within quota limits**.

---

## 6. Common System-Level Failure Modes and Remediation

Below are **real-world failure modes** observed on DGX systems (and HGX-based clusters generally), with diagnosis and remediation strategies. These apply to on-prem DGX; the GCP A3 equivalents are noted where applicable.

### 6.1 GPU Not Detected by Driver

**Symptom:** `nvidia-smi` shows 7 GPUs instead of 8; or `lspci | grep NVIDIA` shows 8 devices but `nvidia-smi` only enumerates 7.

**Root causes:**
- GPU not initialized at boot (firmware hung, PCIe link training failed)
- Driver crash during boot left GPU in bad state
- GPU hardware failure (bad PCIe interface, dead GPU)

**Diagnosis (DGX admin):**
1. Check `dmesg | grep -i nvidia` for driver load errors (e.g., `NVRM: GPU at PCI:0000:8b:00.0 is lost`).
2. Check `lspci -vvv -s <gpu-pci-addr>` for PCIe link speed/width (should be Gen5 x16; if Gen1 x1, link retraining failed).
3. Power-cycle node (cold boot; clears PCIe state).
4. If still missing after reboot, check NVSM for GPU health alerts; if hardware failure suspected, RMA GPU.

**On GCP A3:** If a GPU is missing, the GKE device-plugin will report fewer than 8 GPUs in the node's capacity. Google's monitoring detects this and marks the node for maintenance. Tenant should file support ticket if jobs are scheduled to such a node.

### 6.2 NVLink Fabric Degraded (Partial Link Loss)

**Covered in §3.2.** Key points: check `nvidia-smi nvlink --status`, Fabric Manager logs, error counters; reseat cables / update firmware / reboot; if persistent, RMA.

### 6.3 Training Job Crashes with "NCCL Timeout" or "Illegal Memory Access"

**Symptom:** Multi-GPU training job crashes after N minutes with NCCL timeout (e.g., `NCCL WARN Call to cuMemHostAlloc failed`) or CUDA illegal memory access (e.g., `cudaErrorIllegalAddress`).

**Root causes:**
- Bad GPU memory (uncorrectable ECC error not caught by DCGM diag)
- NVLink data corruption (CRC errors silently corrupting allreduce data)
- Driver bug (rare; usually fixed in latest driver)
- Software bug (NCCL version incompatibility, CUDA OOM due to peak memory spike)

**Diagnosis (DGX admin):**
1. Check `nvidia-smi -q -d MEMORY` for uncorrectable ECC errors (volatile or aggregate). >1 uncorrectable error → RMA GPU.
2. Check `nvidia-smi nvlink --error-counters` for CRC errors. High CRC count on a link → bad cable or link.
3. Run `dcgmi diag -r 3` on the failing GPU. If diagnostic fails, RMA.
4. Enable NCCL debug logs (`NCCL_DEBUG=INFO`) and correlate crash timestamp with NVLink/NCCL warnings.
5. Update driver and NCCL to latest versions.
6. If still crashing, run `nvidia-bug-report.sh` and send to NVIDIA support.

**On GCP A3:** Tenant performs steps 1, 2, 4, 5, 6 (via NCCL debug logs, `nvidia-smi`, `dcgmi diag`, `nvidia-bug-report.sh`). If hardware issue suspected (ECC errors, failed diagnostic), file support ticket; Google replaces node.

### 6.4 System Hangs at Boot (POST / BIOS)

**Symptom:** DGX node powers on but does not reach OS; BMC serial console shows hang at BIOS POST or GPU initialization.

**Root causes:**
- GPU firmware mismatch (InfoROM version incompatible with BIOS)
- NVSwitch firmware hung
- Bad GPU preventing POST (GPU not responding to PCIe config cycles)

**Diagnosis (DGX admin):**
1. Access BMC serial console (`ipmitool sol activate`) to see POST output.
2. If hang is at "Initializing GPUs", suspect GPU firmware or NVSwitch issue.
3. Power-cycle via BMC; if hang persists, enter BIOS setup (F2 during POST) and disable GPU initialization temporarily to reach OS.
4. Once in OS, check GPU firmware versions (`nvsm show firmware`); update if mismatched.
5. If cannot reach OS, boot from DGX recovery ISO and run firmware update from recovery environment.

**On GCP A3:** Tenant has no BMC access. If a node fails to boot, Google's control plane detects it (node never becomes `Ready` in GKE) and marks it for maintenance. Tenant sees the node as `NotReady` but cannot troubleshoot further.

---

## 7. What Tenants CAN Observe on GCP A3 (and What They Cannot)

To summarize the above, here's what a GCP A3 tenant can do today (and what is covered in **lab-11**, when written):

### Tenant CAN:
- **Query GPU inventory**: `nvidia-smi -L`, `kubectl describe node` (see `nvidia.com/gpu: 8`)
- **Monitor GPU health**: `nvidia-smi -q` (temp, power, ECC, clocks, throttle reasons)
- **Check NVLink topology**: `nvidia-smi topo -m` (see full 8-GPU NVLink mesh via NVSwitch)
- **Query NVLink status**: `nvidia-smi nvlink --status` (see which links are UP/DOWN)
- **Check NVLink error counters**: `nvidia-smi nvlink --error-counters` (detect CRC errors, replay errors)
- **Run DCGM diagnostics**: `dcgmi diag -r 1/2/3` (stress test GPUs, memory, peer-to-peer bandwidth)
- **Benchmark NVLink bandwidth**: `nvbandwidth` or `nccl-tests` (measure GPU-GPU communication performance)
- **Capture nvidia-bug-report**: `nvidia-bug-report.sh` (collect driver logs, GPU state, kernel logs; send to GCP support)

### Tenant CANNOT:
- **Access Fabric Manager logs**: `/var/log/fabricmanager.log` not accessible; cannot diagnose NVSwitch training failures
- **Run NVSM**: `nvsm` binary not present; cannot query system-level health, firmware versions, or cooling state
- **Update firmware**: GPU, NVSwitch, NIC firmware updates managed by Google
- **Access BMC**: no IPMI/Redfish access; cannot power-cycle, view serial console, or query thermal sensors
- **Query NVSwitch health directly**: cannot enumerate NVSwitch devices or query per-switch error counters (only GPU-side NVLink counters visible)

**Lab-11 scope:** Lab-11 demonstrates the **tenant-visible** capabilities (nvidia-smi topo, NVLink status, DCGM diagnostics, nvbandwidth) and interprets them in light of the **system-level architecture** taught in this doc (HGX baseboard, Fabric Manager role, NVSM abstraction). Lab-11 does NOT fabricate DGX-specific command outputs; it shows **what is real on our A3 cluster** and explains **what would differ on a DGX system**.

---

## 8. Summary

**DGX H100/H200 systems** are NVIDIA's turnkey AI servers: HGX H100/H200 baseboard (8 GPUs + NVSwitch) + dual Xeon CPUs + ConnectX-7 NICs + NVMe storage + liquid cooling + DGX OS + NVSM + Fabric Manager + BMC. They are fully integrated, validated, and managed via NVSM (system health) and Fabric Manager (NVLink/NVSwitch health). On-prem DGX admins have full access to firmware updates, fabric diagnostics, BMC out-of-band management, and system-level troubleshooting (NVSwitch logs, thermal sensors, power cycling).

**GCP A3 nodes** use the **same HGX H100 baseboard** but abstract the system management layer — Google manages DGX OS / Fabric Manager / NVSM / BMC at the hypervisor/control-plane level. Tenants retain **full GPU-level observability** (`nvidia-smi`, DCGM, NVLink topology, error counters) but lose **system-level control** (no firmware updates, no Fabric Manager logs, no BMC). The **troubleshooting methodology** is the same at the GPU/NVLink level (check ECC, NVLink status, DCGM diag), but system-level issues (fabric training failures, thermal management, firmware bugs) are escalated to GCP support rather than handled by the tenant.

Both platforms deliver **identical GPU compute and NVLink performance** (same hardware, same driver/CUDA stack). The difference is **operational model**: DGX is hands-on, admin-controlled; GCP is managed, tenant-observes-and-reports. For most AI workloads, the tenant-visible capabilities on GCP A3 are sufficient for debugging training crashes, optimizing NCCL collectives, and validating hardware health. For issues requiring Fabric Manager logs or firmware changes, the tenant relies on Google's SRE team (who have DGX-equivalent tooling).

**Practice / observe-and-compare →** lab-11 (when written) will demonstrate the tenant-visible GPU and NVLink observability on our A3 cluster, contrasting with the reference DGX troubleshooting workflows taught here.

---

## References

- **NVIDIA DGX H100 System Architecture Whitepaper**: [https://resources.nvidia.com/en-us-dgx-systems](https://resources.nvidia.com/en-us-dgx-systems)
- **NVIDIA HGX H100 Reference Architecture**: [https://www.nvidia.com/en-us/data-center/hgx/](https://www.nvidia.com/en-us/data-center/hgx/)
- **GPU Fabric Manager Documentation**: [https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/)
- **NVIDIA System Management Interface (NVSM)**: [https://docs.nvidia.com/dgx/](https://docs.nvidia.com/dgx/) (DGX System User Guide → NVSM chapter)
- **Base Command Manager**: [https://docs.nvidia.com/base-command-manager/](https://docs.nvidia.com/base-command-manager/)
- **Portability Matrix (this guide)**: [T6: Portability Matrix](../toolkit/T6-portability-matrix.md) — GCP ↔ NVIDIA product mapping, cross-platform tool equivalents
- **Intra-node NVLink/NVSwitch architecture**: Part I, section 04 (HGX baseboard deep-dive; doc TBD)
- **Health Diagnostics Toolkit**: [T2: Health Diagnostics](../toolkit/T2-health-diagnostics.md) — DCGM diag, XID codes, `nvidia-bug-report.sh`
- **Monitoring Toolkit**: [T1: Monitoring & Inventory](../toolkit/T1-monitoring-inventory.md) — `nvidia-smi`, DCGM, NVML, topology queries
