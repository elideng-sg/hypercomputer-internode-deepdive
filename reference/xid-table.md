# XID Error Reference Table

XID errors are diagnostic codes emitted by the NVIDIA driver to the kernel log when a GPU fault or anomaly is detected. This reference table maps XID codes to their meanings, typical causes, and recommended actions.

The table will be populated with verified XID codes as they are encountered and documented in the labs, particularly in Lab 02 (driver/CUDA health and diagnostics).

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
- XIDs are logged to kernel ring buffer (`dmesg | grep NVRM`), syslog (`/var/log/syslog`), and DCGM (`dcgmi dmon -e`).
- On GKE Container-Optimized OS, `dmesg` may be restricted in containers; use `kubectl get events` or node-level debugging pods for XID visibility.
- This table is populated based on NVIDIA driver documentation (NVML/DCGM reference). Lab 02 verified the diagnostic tooling; Lab 10 will demonstrate fleet-scale XID monitoring.
