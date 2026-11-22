"""Measure warmed scorer latency with and without a concurrent replay update."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from threading import Event
from time import perf_counter

from run import (ROOT, REPLAY, SEED, arrays, frames, original, score,
                 shift, standardize, update)


def timing(rows, stop=None):
    latency = []
    while stop is None or not stop.is_set():
        request = rows[len(latency) % len(rows)]
        started = perf_counter()
        score(original, request, True)
        latency.append((perf_counter() - started) * 1000)
        if stop is None and len(latency) == 1000:
            break
    return latency


def summarize(samples):
    return {
        "requests": len(samples),
        "p50_ms": float(np.percentile(samples, 50)),
        "p90_ms": float(np.percentile(samples, 90)),
    }


def experiment():
    rng = np.random.default_rng(SEED)
    normal = arrays["train"][frames["train"].anomaly.to_numpy() == 0]
    indices = rng.choice(len(normal), REPLAY + 345, replace=False)
    replay, incoming = normal[indices[:REPLAY]], shift(normal[indices[REPLAY:]], 1.75)
    rows = standardize(arrays["test"][:1000])[:, None, :]
    update(original, incoming[:16], replay, SEED)  # Warm optimizer initialization.
    timing(rows)
    idle = timing(rows)
    concurrent = []
    for trial in range(3):
        started = Event()
        done = Event()

        def worker():
            started.wait()
            try:
                update(original, incoming, replay, SEED + trial)
            finally:
                done.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker)
            started.set()
            concurrent.extend(timing(rows, done))
            future.result()
    return {"idle": summarize(idle), "concurrent": summarize(concurrent),
            "concurrent_trials": 3, "scope": "warmed model-only scoring of pre-scaled rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/inference_overlap.json")
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(experiment(), indent=2) + "\n")
