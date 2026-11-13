"""Local HTTP reference deployment for the hybrid wine detector."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "milestone2"))
from demo import Screener
from models import autoencoder

FIELDS = ("transaction_id", "customer_id", "timestamp_seconds", "unit_price",
          "quantity", "amount", "merchant_category", "latitude", "longitude",
          "sample_id", "declared_cultivar")


class Deployment:
    def __init__(self):
        self.screen = Screener()
        self.models = {"original": (self.screen.model, self.screen.threshold)}
        artifact = torch.load(ROOT / "results/adapted.pt", map_location="cpu", weights_only=True)
        candidate = autoencoder(9, self.screen.bundle["bottleneck"])
        candidate.load_state_dict(artifact["state"])
        if (artifact["scaler_mean"] != self.screen.mean.tolist()
                or artifact["scaler_scale"] != self.screen.scale.tolist()):
            raise ValueError("Candidate scaler mismatch")
        self.models["adapted"] = candidate.eval(), artifact["threshold"]
        self.active = "original"
        self.latest = {}
        self.completed = {}

    def process(self, path, body, queued_at):
        wait = (perf_counter() - queued_at) * 1000
        if path == "/activate":
            if body.get("version") not in self.models:
                raise ValueError("Unknown model version")
            self.active = body["version"]
            return {"version": self.active, "queue_ms": wait}
        if set(body) != set(FIELDS):
            raise ValueError("Invalid transaction fields")
        for key in ("transaction_id", "customer_id"):
            if not isinstance(body[key], str) or not 0 < len(body[key]) <= 128:
                raise ValueError("Invalid identifier")
        for key in ("merchant_category", "sample_id", "declared_cultivar", "quantity"):
            if type(body[key]) is not int:
                raise ValueError("Invalid integer field")
        for key in ("timestamp_seconds", "unit_price", "amount", "latitude", "longitude"):
            if type(body[key]) not in (int, float) or not np.isfinite(body[key]):
                raise ValueError("Invalid numeric field")
        if min(body[key] for key in ("timestamp_seconds", "amount", "unit_price", "quantity", "merchant_category")) < 0:
            raise ValueError("Negative transaction value")
        tid, cid, stamp = (body[key] for key in ("transaction_id", "customer_id", "timestamp_seconds"))
        if tid in self.completed:
            original, response = self.completed[tid]
            if body != original:
                raise ValueError("Conflicting transaction ID")
            return {**response, "duplicate": True, "queue_ms": wait}
        if cid in self.latest and stamp <= self.latest[cid]:
            raise ValueError("Out-of-order customer event")
        self.screen.model, self.screen.threshold = self.models[self.active]
        result = self.screen.predict_event(body)
        self.latest[cid] = stamp
        response = {**result, "version": self.active, "queue_ms": wait}
        self.completed[tid] = body, response
        return response


class Server(ThreadingHTTPServer):
    request_queue_size = 64

    def __init__(self, address, slots):
        self.deployment = Deployment()
        self.capacity = BoundedSemaphore(slots)
        self.worker = ThreadPoolExecutor(max_workers=1)
        super().__init__(address, Handler)

    def server_close(self):
        super().server_close()
        self.worker.shutdown()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path not in ("/score", "/activate"):
            self.send_json(404, {"error": "Unknown route"})
            return
        self.connection.settimeout(2)
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
        except (ValueError, TimeoutError) as error:
            self.send_json(400, {"error": str(error)})
            return
        if not self.server.capacity.acquire(blocking=False):
            self.send_json(503, {"error": "Scoring queue full"})
            return
        try:
            result = self.server.worker.submit(
                self.server.deployment.process, self.path, payload, perf_counter()).result()
        except (ValueError, KeyError, TypeError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception:
            self.send_json(500, {"error": "Scoring failed"})
        else:
            self.send_json(200, result)
        finally:
            self.server.capacity.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    with Server(("127.0.0.1", args.port), slots=16) as server:
        server.serve_forever()
