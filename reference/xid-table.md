# XID Error Reference Table

XID errors are diagnostic codes emitted by the NVIDIA driver to the kernel log when a GPU fault or anomaly is detected. This reference table maps XID codes to their meanings, typical causes, and recommended actions.

It is the decode target for **[lab-14](../labs/lab-14-single-gpu-health-triage/)** (single-GPU health triage) and **[doc-17](../docs/part5-operations-diagnostics/17-single-gpu-node-health.md)** (single-GPU & node health), which show **where XIDs actually surface on managed GKE** — Node Problem Detector → Cloud Logging — since unprivileged containers can't read `dmesg` **and** the managed `dcgm-exporter` does **not** export `DCGM_FI_DEV_XID_ERRORS` by default (confirmed live in lab-14; see notes below).

---

| XID | Meaning | Typical cause | Action |
| :--- | :--- | :--- | :--- |
| **13** | Graphics Engine Exception | Invalid GPU operation, corrupted command buffer, driver bug | Check for driver/firmware mismatch, review application logs, consider driver update or `nvidia-bug-report.sh` |
| **31** | GPU Memory Page Fault | Application accessed invalid GPU memory address | Check application for out-of-bounds access, review CUDA error handling, verify memory allocations |
| **43** | GPU Stopped Responding | GPU hung during execution (timeout) | Check for infinite loop in kernel, thermal throttle (`nvidia-smi -q -d PERFORMANCE`), or power issue; may require GPU reset |
| **48** | Double Bit ECC Error (Uncorrectable) | DRAM defect, cosmic ray, severe memory corruption | **Critical.** Check ECC counters (`nvidia-smi -q -d ECC`), review page retirement status; RMA GPU if errors recur |
| **63** | Row Remapping Event | GPU firmware remapped a failing DRAM row | Normal if infrequent; monitor remap count via `nvidia-smi -q -d ROW_REMAP`. Indicates localized DRAM defect being corrected. |
| **64** | Row Remapping Failure | GPU ran out of spare rows or remapping failed | **Fatal.** Check row remap status (`nvidia-smi -q -d ROW_REMAP`); GPU likely requires RMA |
| **74** | NVLink Error | NVLink CRC error, protocol error, or link training failure | Check NVLink topology (`nvidia-smi nvlink -s`), Fabric Manager status, cable seating; re-train links or reboot node |
| **79** | GPU Fallen Off the Bus | PCIe link lost, GPU no longer enumerable | **Fatal.** Check PCIe link status (`nvidia-smi -q -d PCIE`), reseat GPU, inspect for hardware fault (power, cable, riser); may require RMA |
| **94** | Contained ECC Error | ECC error within a GPU context (application isolated) | Error did not propagate to kernel; check application robustness, monitor ECC rate. Less severe than XID 95. |
| **95** | Uncontained ECC Error | ECC error propagated to system (may affect kernel or other processes) | **Severe.** Check for memory corruption, verify system stability, RMA GPU if errors continue |

**Notes:**
- On bare metal / DGX, XIDs land in the kernel ring buffer (`dmesg | grep NVRM`), syslog (`/var/log/syslog`), and DCGM (`dcgmi dmon -e 230` → `DCGM_FI_DEV_XID_ERRORS`).
- **On managed GKE (Container-Optimized OS) containers cannot read `dmesg`** — that restriction is itself the lesson. XIDs surface instead via **Node Problem Detector → Cloud Logging** (query the node's log for `NVRM: Xid`), and `kubectl get events` on the node catches the NPD-raised condition. **Correction (confirmed live in lab-14, gotcha G11):** the **managed** `dcgm-exporter` (lab-10 fleet pipeline) does **not** export `DCGM_FI_DEV_XID_ERRORS` — its default field set is 19 FB/temp/util/power/clock + PROF_* metrics, XID not among them. To get the XID *metric* you must run your **own** exporter with a custom field list including `DCGM_FI_DEV_XID_ERRORS`; otherwise NPD → Cloud Logging is the path. lab-14 walks this GKE-native path end to end (and reads the contrast: a *privileged* pod **can** read `dmesg`, but production tooling should not depend on that).
- Severity shorthand above (**Critical/Fatal/Severe**) follows the NVIDIA driver (NVML/DCGM) reference. Codes **48/63/64/79/94/95** are the hardware-fault family that is **unsafe to inject live** on held Flex capacity — lab-14 decodes them from **curated, clearly-labeled** captured signatures, while throttle/thermal (a benign, reversible load) is reproduced for real.
