#!/usr/bin/env python3
"""All-reduce bandwidth sweep over torch.distributed (NCCL backend).

Launched by torchrun with one process per GPU across one or more nodes.
Reports, per message size, the algorithm bandwidth (algbw) and the bus
bandwidth (busbw) using the standard ring all-reduce correction:

    busbw = algbw * 2*(n-1)/n

This is the same figure nccl-tests' all_reduce_perf reports, computed here
via the identical NCCL library so it runs cleanly across two Kubernetes pods
without an MPI/SSH launcher. Sizes and iteration counts mirror a typical
nccl-tests sweep (`-b 8 -e 8G -f 2`).
"""
import os
import time
import torch
import torch.distributed as dist


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f}{unit}" if nbytes >= 1 else f"{nbytes}B"
        nbytes /= 1024


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    warmup = int(os.environ.get("BENCH_WARMUP", "5"))
    iters = int(os.environ.get("BENCH_ITERS", "20"))
    # 8 B .. 2^MAX_EXP bytes, doubling. Default 8 GiB (33); lower for slower
    # inter-node fabrics to bound wall-clock.
    max_exp = int(os.environ.get("BENCH_MAX_EXP", "33"))
    sizes = [2 ** e for e in range(3, max_exp + 1)]

    if rank == 0:
        print(f"# world_size={world}  backend=nccl  warmup={warmup} iters={iters}")
        print(f"# {'size':>10} {'count':>12} {'time_ms':>10} "
              f"{'algbw_GBps':>12} {'busbw_GBps':>12}")

    for nbytes in sizes:
        count = nbytes // 4  # float32 elements
        if count == 0:
            continue
        try:
            buf = torch.ones(count, dtype=torch.float32, device="cuda")
        except RuntimeError:
            if rank == 0:
                print(f"# stop: OOM at {human(nbytes)}")
            break

        for _ in range(warmup):
            dist.all_reduce(buf)
        torch.cuda.synchronize()

        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        # algbw = data moved / time; busbw applies the ring all-reduce factor.
        algbw = nbytes / dt / 1e9
        busbw = algbw * 2 * (world - 1) / world
        if rank == 0:
            print(f"  {human(nbytes):>10} {count:>12} {dt*1e3:>10.3f} "
                  f"{algbw:>12.2f} {busbw:>12.2f}")
        del buf
        torch.cuda.empty_cache()

    dist.barrier()
    if rank == 0:
        print("# done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
