#!/usr/bin/env bash
set -euo pipefail

# Source capture helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/lib_capture.sh"

LAB_NAME="lab-03"
WORK_DIR="/work"

echo "=== Lab 03: Single-GPU Benchmark & Profile ==="
echo "Repo root: $REPO_ROOT"
echo "Assets: $ASSETS/$LAB_NAME"

# Ensure we're running inside the gpu-debug pod
if [ ! -f /usr/local/bin/nvcc ]; then
  echo "ERROR: Not running inside CUDA container; nvcc not found"
  exit 1
fi

# Step 1: nvbandwidth (H2D/D2H)
echo ""
echo "Step 1: Running nvbandwidth for H2D/D2H bandwidth..."

# Check if nvbandwidth is available; if not, try to use bandwidthTest from CUDA samples
if command -v nvbandwidth &> /dev/null; then
  cap_run "$LAB_NAME" "nvbandwidth.txt" -- nvbandwidth -t testcase=1,testcase=2 || true
else
  echo "nvbandwidth not found; attempting to build from source or use bandwidthTest..."

  # Try bandwidthTest if available
  if command -v bandwidthTest &> /dev/null; then
    cap_run "$LAB_NAME" "nvbandwidth.txt" -- bandwidthTest --htod --dtoh
  else
    # Try to build nvbandwidth from source
    if [ -d /opt/nvbandwidth ]; then
      cd /opt/nvbandwidth
      make -j || true
      if [ -f ./nvbandwidth ]; then
        cap_run "$LAB_NAME" "nvbandwidth.txt" -- ./nvbandwidth -t testcase=1,testcase=2 || true
      fi
    else
      # Clone and build
      cd /tmp
      git clone https://github.com/NVIDIA/nvbandwidth.git || true
      if [ -d nvbandwidth ]; then
        cd nvbandwidth
        make -j || true
        if [ -f ./nvbandwidth ]; then
          cap_run "$LAB_NAME" "nvbandwidth.txt" -- ./nvbandwidth -t testcase=1,testcase=2 || true
        fi
      fi
    fi
  fi

  # If still no output, use note that only bandwidthTest was available
  if [ ! -s "$ASSETS/$LAB_NAME/nvbandwidth.txt" ]; then
    echo "# Note: nvbandwidth not available; used available tool or skipped" | tee "$ASSETS/$LAB_NAME/nvbandwidth.txt"
  fi
fi

# Step 2: Run gemm.py to generate gemm.csv
echo ""
echo "Step 2: Running GEMM benchmark..."
python3 "$SCRIPT_DIR/gemm.py" | tee "$ASSETS/$LAB_NAME/gemm.csv"

# Step 3: nsys profile
echo ""
echo "Step 3: Running nsys profile..."
cd "$WORK_DIR"
nsys profile -t cuda,nvtx -o "$WORK_DIR/gemm" python3 "$SCRIPT_DIR/gemm.py" > /dev/null 2>&1

# Generate nsys stats
nsys stats "$WORK_DIR/gemm.nsys-rep" > "$ASSETS/$LAB_NAME/nsys-stats.txt" 2>&1 || true

# Copy the .nsys-rep file out (for later GUI inspection if needed)
echo "Copying gemm.nsys-rep to assets..."
cp "$WORK_DIR/gemm.nsys-rep" "$ASSETS/$LAB_NAME/" || true

# Step 4: Attempt ncu
echo ""
echo "Step 4: Attempting ncu (Nsight Compute) profile..."
ncu --set full -o "$WORK_DIR/gemm_ncu" python3 "$SCRIPT_DIR/gemm.py" > "$ASSETS/$LAB_NAME/ncu-output.txt" 2>&1 || {
  echo "ncu failed (expected on GKE COS without CAP_SYS_ADMIN)"
  # Capture the error
  cat "$ASSETS/$LAB_NAME/ncu-output.txt" || true
}

# Check if ncu succeeded (by checking if .ncu-rep file exists)
if [ -f "$WORK_DIR/gemm_ncu.ncu-rep" ]; then
  echo "ncu succeeded; copying output..."
  cp "$WORK_DIR/gemm_ncu.ncu-rep" "$ASSETS/$LAB_NAME/" || true
else
  echo "ncu failed; error captured in ncu-output.txt"
fi

# Provenance
echo ""
echo "Recording provenance..."
NODES=$(cap_nodes | tr '\n' ' ' | sed 's/ *$//')
cap_verify_provenance "$LAB_NAME" "gemm.csv,nsys-stats.txt,nvbandwidth.txt" "$NODES" "single-GPU-profile"

echo ""
echo "=== Lab 03 Complete ==="
echo "Artifacts written to: $ASSETS/$LAB_NAME/"
ls -lh "$ASSETS/$LAB_NAME/"
