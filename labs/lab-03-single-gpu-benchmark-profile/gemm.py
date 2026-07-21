#!/usr/bin/env python3
"""
Single-GPU GEMM benchmark with NVTX ranges and timing.
Runs square matrix multiplications (M=N=K) across multiple sizes and dtypes.
Prints CSV with dtype,size,tflops for parsing.
"""
import torch
import sys

SIZES = [2048, 4096, 8192, 16384]
WARMUP_ITERS = 5
TIMED_ITERS = 20

def benchmark_gemm(size, dtype):
    """Run a square GEMM (MxM @ MxM) and return TFLOPs."""
    device = torch.device("cuda:0")

    # Allocate matrices
    A = torch.randn(size, size, dtype=dtype, device=device)
    B = torch.randn(size, size, dtype=dtype, device=device)

    # Warmup
    for _ in range(WARMUP_ITERS):
        C = torch.mm(A, B)
    torch.cuda.synchronize()

    # Timed runs with NVTX range and CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.nvtx.range_push(f"GEMM_{dtype}_{size}")
    start_event.record()
    for _ in range(TIMED_ITERS):
        C = torch.mm(A, B)
    end_event.record()
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s = elapsed_ms / 1000.0

    # Compute TFLOPs: 2*M*N*K FLOPs per GEMM, M=N=K=size
    flops = 2 * size * size * size * TIMED_ITERS
    tflops = flops / elapsed_s / 1e12

    return tflops

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    print("dtype,size,tflops")

    # FP16
    for size in SIZES:
        tflops = benchmark_gemm(size, torch.float16)
        print(f"fp16,{size},{tflops:.2f}")

    # BF16
    for size in SIZES:
        tflops = benchmark_gemm(size, torch.bfloat16)
        print(f"bf16,{size},{tflops:.2f}")

    # FP8 (if supported on H100)
    # PyTorch 2.1+ supports torch.float8_e4m3fn and torch.float8_e5m2
    # Check if available; if not, skip
    try:
        # Check if FP8 dtypes exist
        fp8_dtype = torch.float8_e4m3fn
        # Try a small allocation to verify support
        test = torch.zeros(2, 2, dtype=fp8_dtype, device="cuda:0")

        for size in SIZES:
            tflops = benchmark_gemm(size, fp8_dtype)
            print(f"fp8,{size},{tflops:.2f}")
    except (AttributeError, RuntimeError):
        # FP8 not supported; skip
        print("# FP8 not supported, skipping", file=sys.stderr)

if __name__ == "__main__":
    main()
