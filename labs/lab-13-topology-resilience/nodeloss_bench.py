#!/usr/bin/env python3
"""Long-running all-reduce loop for job-level node-loss injection (lab-13b).

Launched exactly like allreduce_bench.py (one process per GPU via launch_node.sh),
but instead of a finite sweep it loops a fixed-size all-reduce forever, printing a
per-iteration heartbeat. This lets an operator kill the ranks on ONE node mid-run
and capture what the *surviving* ranks do.

Two deliberate settings make the fault DIAGNOSABLE instead of a silent 30-minute
hang (the NCCL default):
  * init_process_group(timeout=PG_TIMEOUT) — bounds how long a collective blocks
    waiting for a vanished peer before the process group aborts.
  * TORCH_NCCL_ASYNC_ERROR_HANDLING=1 (set by the runner) — the NCCL watchdog
    tears the process down with a logged error the instant a peer connection
    breaks, rather than deadlocking.

On any collective failure the rank prints a single "# FAULT ..." line (the fault
signature) and exits non-zero. WHY 3 NODES: killing one of two nodes leaves a
single-node remnant that is no longer a distributed job — there is no *surviving
multi-node set* to observe or reschedule. At three nodes, killing one leaves a
genuine 16-GPU / 2-node survivor set (lab-13b's rerun step).
"""
import datetime
import os
import sys
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    node = os.environ.get("NODE_NAME", "?")
    pg_timeout = int(os.environ.get("PG_TIMEOUT", "90"))
    # ~256 MiB fp32 — large enough that each iter genuinely crosses the fabric,
    # small enough to loop at a readable cadence.
    count = int(os.environ.get("NODELOSS_ELEMS", str(64 * 1024 * 1024)))
    heartbeat_every = int(os.environ.get("NODELOSS_HEARTBEAT", "5"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(seconds=pg_timeout)
    )
    buf = torch.ones(count, dtype=torch.float32, device="cuda")

    if rank == 0:
        print(f"# nodeloss: world_size={world} pg_timeout={pg_timeout}s "
              f"elems={count} ({count*4/1e6:.0f} MB fp32)", flush=True)

    it = 0
    t0 = time.perf_counter()
    try:
        while True:
            dist.all_reduce(buf)
            torch.cuda.synchronize()
            it += 1
            # One heartbeat per node (its local rank 0) so the log shows which
            # nodes are still alive over time.
            if local_rank == 0 and it % heartbeat_every == 0:
                print(f"# heartbeat rank={rank} node={node} iter={it} "
                      f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
            time.sleep(0.2)
    except Exception as exc:  # noqa: BLE001 — we want the raw signature
        print(f"# FAULT rank={rank} node={node} iter={it}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        sys.stdout.flush()
        # Non-zero so launch_node.sh propagates rc=1 for the survivor set.
        os._exit(42)


if __name__ == "__main__":
    main()
