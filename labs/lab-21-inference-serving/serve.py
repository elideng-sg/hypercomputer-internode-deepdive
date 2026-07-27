#!/usr/bin/env python3
# serve.py — a minimal but REAL GPU inference server, to make the serving
# workload measurable. It serves ResNet-50 on cuda:0 with SERVER-SIDE DYNAMIC
# BATCHING — the single most important serving knob — so the lab can show the
# throughput/latency tradeoff that defines inference (vs. the throughput-only
# world of training in lab-20).
#
# A background worker thread pulls up to BATCH_MAX in-flight requests (waiting at
# most BATCH_DELAY_MS to fill a batch), runs them as ONE forward pass, and wakes
# each caller with its latency + the batch size it was served in. Weights are
# random (weights=None) — inference latency is identical to a trained model and
# needs no download.
#
# env: PORT(8080), BATCH_MAX(16), BATCH_DELAY_MS(5), DEVICE(cuda:0)
import os, time, threading, queue, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch, torchvision

PORT          = int(os.environ.get("PORT", "8080"))
BATCH_MAX     = int(os.environ.get("BATCH_MAX", "16"))
BATCH_DELAY_S = float(os.environ.get("BATCH_DELAY_MS", "5")) / 1000.0
DEVICE        = os.environ.get("DEVICE", "cuda:0")

torch.backends.cudnn.benchmark = True
dev = torch.device(DEVICE)
model = torchvision.models.resnet50(weights=None).eval().half().to(dev, memory_format=torch.channels_last)
# a single resident input template; each "request" is one image of it
IMG = torch.randn(1, 3, 224, 224, device=dev).half().contiguous(memory_format=torch.channels_last)

# warm up (CUDA context, cudnn autotune) so the first real request isn't an outlier
with torch.no_grad():
    for b in (1, BATCH_MAX):
        _ = model(IMG.repeat(b, 1, 1, 1)); torch.cuda.synchronize()

Q = queue.Queue()

def batch_worker():
    while True:
        item = Q.get()                          # (result_list, event) ; blocks
        batch = [item]
        deadline = time.time() + BATCH_DELAY_S
        while len(batch) < BATCH_MAX:
            timeout = deadline - time.time()
            if timeout <= 0: break
            try: batch.append(Q.get(timeout=timeout))
            except queue.Empty: break
        bs = len(batch)
        t0 = time.time()
        with torch.no_grad():
            _ = model(IMG.repeat(bs, 1, 1, 1)); torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000.0
        for res, ev in batch:
            res["batch"] = bs; res["gpu_ms"] = round(dt, 2); ev.set()

threading.Thread(target=batch_worker, daemon=True).start()

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/infer"):
            res = {}; ev = threading.Event(); t0 = time.time()
            Q.put((res, ev)); ev.wait(30)
            res["latency_ms"] = round((time.time() - t0) * 1000.0, 2)
            body = json.dumps(res).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers()
            self.wfile.write(b"ok")

print(f"[serve] ResNet-50 on {DEVICE}  BATCH_MAX={BATCH_MAX} BATCH_DELAY_MS={BATCH_DELAY_S*1000:.0f}  :{PORT}", flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
