#!/usr/bin/env python3
# loadgen.py — a closed-loop load generator for serve.py. C concurrent clients
# each fire /infer back-to-back for DURATION seconds; we report achieved
# throughput (req/s) and latency percentiles. Sweeping C traces the SERVING
# SATURATION KNEE: throughput rises with concurrency until the GPU is full, then
# latency climbs while throughput plateaus — the curve an autoscaler reacts to.
#
# env: URL(http://127.0.0.1:8080/infer), CONC(8), DURATION(10)
import os, time, threading, urllib.request

URL      = os.environ.get("URL", "http://127.0.0.1:8080/infer")
CONC     = int(os.environ.get("CONC", "8"))
DURATION = float(os.environ.get("DURATION", "10"))

lats = []; lock = threading.Lock(); stop = time.time() + DURATION

def client():
    local = []
    while time.time() < stop:
        t0 = time.time()
        try:
            urllib.request.urlopen(URL, timeout=30).read()
            local.append((time.time() - t0) * 1000.0)
        except Exception:
            pass
    with lock: lats.extend(local)

# brief warm connection, then timed run
threads = [threading.Thread(target=client) for _ in range(CONC)]
t_start = time.time()
for t in threads: t.start()
for t in threads: t.join()
wall = time.time() - t_start

lats.sort(); n = len(lats)
def pct(p): return lats[min(n - 1, int(n * p / 100.0))] if n else 0.0
qps = n / wall if wall else 0.0
print(f"CONC={CONC:3d}  n={n:6d}  {qps:8.1f} req/s  "
      f"p50={pct(50):7.1f}ms  p95={pct(95):7.1f}ms  p99={pct(99):7.1f}ms  "
      f"mean={sum(lats)/n if n else 0:7.1f}ms", flush=True)
