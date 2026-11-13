"""Exercise full HTTP scoring, queue backpressure and rollback on synthetic events."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
import pandas as pd

from service import FIELDS, ROOT, Server

events = pd.read_csv(ROOT.parent / "milestone2/data/events_test.csv")
customers = [[{key: row[key] for key in FIELDS} for row in group.to_dict("records")]
             for _, group in events.groupby("customer_id")][:32]


def post(server, path, value):
    request = Request(f"http://127.0.0.1:{server.server_port}{path}",
                      json.dumps(value).encode(), {"Content-Type": "application/json"})
    start = perf_counter()
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
            return response.status, json.load(response), (perf_counter() - start) * 1000
    except HTTPError as error:
        return error.code, json.load(error), (perf_counter() - start) * 1000


def experiment(concurrency):
    with Server(("127.0.0.1", 0), slots=16) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def customer(records):
                return [post(server, "/score", record) for record in records]

            for records in customers:
                assert all(status == 200 and body["action"] == "history_warmup"
                           for status, body, _ in customer(records[:5]))
            start = perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = [result for batch in pool.map(customer, [c[5:] for c in customers])
                           for result in batch]
            elapsed = perf_counter() - start
        finally:
            server.shutdown()
            thread.join()
    scored = [r for r in results if r[0] == 200 and r[1]["action"] != "history_warmup"]
    return {"clients": concurrency, "requests": len(results), "scored": len(scored),
            "errors": len(results) - len(scored), "scored_per_second": len(scored) / elapsed,
            "p50_ms": float(np.percentile([r[2] for r in scored], 50)),
            "p90_ms": float(np.percentile([r[2] for r in scored], 90)),
            "queue_p90_ms": float(np.percentile([r[1]["queue_ms"] for r in scored], 90))}


def verify_controls():
    with Server(("127.0.0.1", 0), slots=1) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        release = Event()
        entered = Event()
        process = server.deployment.process

        def blocked(*args):
            entered.set()
            release.wait(timeout=5)
            return process(*args)

        try:
            rows = customers[0]
            for row in rows[:6]:
                assert post(server, "/score", row)[0] == 200
            assert post(server, "/score", rows[5])[1]["duplicate"]
            assert post(server, "/score", {**rows[5], "transaction_id": "late"})[0] == 400
            assert post(server, "/activate", {"version": "adapted"})[0] == 200
            candidate = post(server, "/score", rows[6])[1]
            start = perf_counter()
            assert post(server, "/activate", {"version": "original"})[0] == 200
            restored = post(server, "/score", rows[7])[1]
            rollback_ms = (perf_counter() - start) * 1000
            assert candidate["version"] == "adapted" and restored["version"] == "original"
            reference = Server(("127.0.0.1", 0), slots=1)
            try:
                for row in rows[:8]:
                    expected = reference.deployment.process("/score", row, perf_counter())
                assert restored["anomaly_score"] == expected["anomaly_score"]
            finally:
                reference.server_close()
            server.deployment.process = blocked
            with ThreadPoolExecutor(max_workers=1) as pool:
                request = pool.submit(post, server, "/score", rows[8])
                assert entered.wait(timeout=3)
                try:
                    assert post(server, "/score", customers[1][0])[0] == 503
                finally:
                    release.set()
                assert request.result()[0] == 200
            assert post(server, "/score", customers[1][0])[0] == 200
            return {"rollback_and_next_score_ms": rollback_ms, "restored_score_matches": True,
                    "queue_full_status": 503, "queue_recovery_status": 200,
                    "duplicate_idempotent": True, "out_of_order_status": 400}
        finally:
            release.set()
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    results = {"load": [experiment(n) for n in (1, 4, 16)], "controls": verify_controls()}
    destination = Path(ROOT / "results/deployment.json")
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
