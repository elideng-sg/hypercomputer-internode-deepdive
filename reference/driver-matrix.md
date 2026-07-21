# GPU Driver Version Matrix

This reference provides the recommended GPU driver versions for each GCP machine type and GPU model. The matrix below is copied from verified GCP documentation and will be re-verified against live `nvidia-smi` output from the lab cluster in Task 2.2.

---

## Recommended GPU Driver Matrix

| Machine Type | GPU Model | Supported Branches (NVIDIA) | Recommended Branch | End of Support (Recommended) | Min Driver Version (Linux) | Min Driver Version (Windows) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A4X Max** | Blackwell GB300 | R580 or later | R580 | June 2028 | 580.95.05 | N/A |
| **A4X** | Blackwell GB200 | R580 or later | R580 | June 2028 | 580.95.05 | N/A |
| **A4** | Blackwell B200 | R580 or later | R580 | June 2028 | 580.95.05 | N/A |
| **A3 Ultra** | H200 (141GB HBM3e) | R580 or later | R580 | June 2028 | 580.95.05 | N/A |
| **A3 Mega/High/Edge** | H100 (80GB HBM3) | R580 or later | R580 | June 2028 | 580.95.05 | N/A |
| **G4** | RTX PRO 6000 (Full) | R580 or later | R580 | June 2028 | 580.95.05 | 581.42 |
| **G4 (Fractional)** | RTX PRO 6000 | R580 or later | R580 | June 2028 | 580.126.09 | 582.16 |
| **G2** | L4 | R580 or later | R580 | June 2028 | 580.95.05 | 581.42 |
| **A2 Standard/Ultra** | A100 | R580 or later | R580 | June 2028 | 580.95.05 | 581.42 |
| **N1** | T4 | R580 or later | R580 | June 2028 | 580.95.05 | 581.42 |
| **N1** | V100, P100, P4 | R580 | R580 | June 2028 | 580.95.05 | 581.42 |

**Note:** This matrix will be cross-verified with live `nvidia-smi` output from the A3 H100 lab cluster during Task 2.2 (driver and CUDA health checks).

---

## NVIDIA Driver Branches

NVIDIA provides three driver branches with different support lifecycles:

- **Long-Term Support Branch (LTSB):** Prioritizes stability with minimal changes; extended support lifecycle of **3 years**. Latest GCP-verified LTSB: **R580** (End of support: June 2028).
- **Production Branch (PB):** Focuses on performance optimizations and immediate support for new hardware; shorter support lifecycle of **up to 1 year**. Latest GCP-verified PB: **R595** (End of support: March 2027).
- **New Feature Branch (NFB):** For testing new features only; not recommended for production.

For production workloads on GCP, **LTSB (R580)** is recommended.
