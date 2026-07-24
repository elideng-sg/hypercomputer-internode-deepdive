#!/usr/bin/env python3
"""Small NCCL all-reduce for inter-node comms triage (lab-15).

Unlike allreduce_bench.py (a bandwidth *sweep*), this runs a handful of
fixed-size all-reduces and prints a per-rank ARRIVE marker and per-iter
progress from rank 0. The point is triage, not throughput: when a comms fault
hits, a hung or straggling rank is visible by its *absence* from the markers,
and the timing of the watchdog abort tells you crash-vs-hang.

Two settings make every fault fast and diagnosable instead of a silent hang:
  * init_process_group(timeout=PG_TIMEOUT) bounds how long a collective blocks
    on a peer that never arrives;
  * TORCH_NCCL_ASYNC_ERROR_HANDLING=1 (set by the launcher) tears the process
    down with a logged error the moment a peer socket breaks.

Env (RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT come from launch_node.sh):
  PG_TIMEOUT       seconds to bound a blocked collective            (default 45)
  COMMS_ITERS      number of all-reduces                            (default 20)
  COMMS_NBYTES     buffer size in bytes                             (default 256 MiB)
  STRAGGLER_RANK   if set, this global rank sleeps before iter 0    (default unset)
  STRAGGLER_SLEEP  seconds the straggler sleeps                     (default 0)
"""
import datetime
import os
import time

import torch
import torch.distributed as dist


def ts() -> str:
    return datetime.datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    node = os.environ.get("NODE_NAME", "?")
    pg_timeout = int(os.environ.get("PG_TIMEOUT", "45"))
    iters = int(os.environ.get("COMMS_ITERS", "20"))
    nbytes = int(os.environ.get("COMMS_NBYTES", str(256 * 1024 * 1024)))
    straggler_rank = os.environ.get("STRAGGLER_RANK", "")
    straggler_sleep = int(os.environ.get("STRAGGLER_SLEEP", "0"))

    torch.cuda.set_device(local_rank)
    # ARRIVE is printed BEFORE the first collective, so a rank that never
    # reaches the collective (a straggler) is still visible as "arrived".
    print(f"# ARRIVE rank={rank} node={node} local={local_rank} "
          f"world={world} {ts()}", flush=True)

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=pg_timeout),
    )
    if rank == 0:
        print(f"# comms_bench world={world} iters={iters} "
              f"nbytes={nbytes} pg_timeout={pg_timeout}s {ts()}", flush=True)

    count = nbytes // 4  # float32 elements
    buf = torch.ones(count, dtype=torch.float32, device="cuda")

    # Warmup all_reduce with ALL ranks present: this is what actually builds the
    # NCCL communicator (ncclCommInitRank runs lazily on the first collective).
    # We do it *before* any straggler sleep so the communicator is fully formed —
    # otherwise a late rank stalls comm *init*, which has its own (long) bootstrap
    # timeout, not the PG-work timeout we want to demonstrate.
    dist.all_reduce(buf)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"# warmup done, communicator established {ts()}", flush=True)

    # Straggler injection: with the comm already up, one rank arrives late at the
    # NEXT collective, so the other ranks block in all_reduce and the watchdog
    # aborts them at *exactly* PG_TIMEOUT — the hang signature (distinct from a
    # peer crash, which aborts in seconds via a closed socket).
    if straggler_rank != "" and rank == int(straggler_rank) and straggler_sleep > 0:
        print(f"# STRAGGLER rank={rank} node={node} sleeping {straggler_sleep}s "
              f"before all_reduce {ts()}", flush=True)
        time.sleep(straggler_sleep)

    for it in range(iters):
        buf.fill_(1.0)                 # reset so the reduced value is always == world
        dist.all_reduce(buf)
        torch.cuda.synchronize()
        if rank == 0 and (it % 5 == 0 or it == iters - 1):
            print(f"# iter={it} all_reduce ok {ts()}", flush=True)

    val = buf[0].item()
    dist.barrier()
    if rank == 0:
        print(f"COMMS all_reduce OK value={val} (expect {float(world)}) "
              f"world_size={world} {ts()}", flush=True)
        print("# done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
