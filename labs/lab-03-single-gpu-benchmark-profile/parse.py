#!/usr/bin/env python3
"""
Parse gemm.csv and generate TFLOPs bar chart using lib_plot.py.
"""
import sys
import os
import subprocess
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LIB_PLOT = os.path.join(REPO_ROOT, "scripts/lib_plot.py")
ASSETS_DIR = os.path.join(REPO_ROOT, "assets/lab-03")
GEMM_CSV = os.path.join(ASSETS_DIR, "gemm.csv")
OUTPUT_PNG = os.path.join(ASSETS_DIR, "gemm_tflops.png")

def main():
    if not os.path.exists(GEMM_CSV):
        print(f"ERROR: {GEMM_CSV} not found", file=sys.stderr)
        sys.exit(1)

    # Read CSV
    df = pd.read_csv(GEMM_CSV, comment='#')

    # Pivot: rows = size, columns = dtype
    pivot = df.pivot(index='size', columns='dtype', values='tflops')

    # Write pivoted CSV for lib_plot.py
    pivot_csv = os.path.join(ASSETS_DIR, "gemm_pivot.csv")
    pivot.reset_index().to_csv(pivot_csv, index=False)

    # Determine which dtypes are present
    dtypes_present = pivot.columns.tolist()
    y_cols = ",".join(dtypes_present)

    # Call lib_plot.py
    cmd = [
        "python3", LIB_PLOT,
        "--csv", pivot_csv,
        "--x", "size",
        "--y", y_cols,
        "--out", OUTPUT_PNG,
        "--kind", "bar",
        "--title", "GEMM Performance by Data Type",
        "--xlabel", "Matrix Size (M=N=K)",
        "--ylabel", "TFLOPs"
    ]

    subprocess.run(cmd, check=True)
    print(f"Plot written to {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
