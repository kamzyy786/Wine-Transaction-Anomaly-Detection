"""Replay, simulated review, and held-out evaluation for the wine detector."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from milestone3.monitor import psi
from milestone3.run import (arrays, autoencoder, benchmark, bundle, checkpoint, evaluate, features, frames,
                 hardware, original, rss_mib, score, standardize, threshold)

SEED = 64504
BUDGET = 64
REPLAY = 256
EPOCHS = 8
BATCH = 64
LOCATION, PRICE = 5, 6


def shift(values, strength):
    shifted = values.copy()
    shifted[:, [LOCATION, PRICE]] += np.array([0.45, 0.65]) * strength
    return shifted


def active_query(scores, cutoff, budget):
    # Alternate near-boundary and low-error rows to include likely normal examples.
    boundary = np.argsort(np.abs(scores - cutoff), kind="stable")[:budget // 2]
    remaining = np.setdiff1d(np.arange(len(scores)), boundary)
    likely_normal = remaining[np.argsort(scores[remaining], kind="stable")]
    return np.r_[boundary, likely_normal[:budget - len(boundary)]]


def update(current, normals, replay, seed):
    candidate = copy.deepcopy(current)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=0.001, weight_decay=0.0001)
    data = torch.from_numpy(standardize(np.concatenate((normals, replay))))
    generator = torch.Generator().manual_seed(seed)
    candidate.train()
    for _ in range(EPOCHS):
        for batch in torch.randperm(len(data), generator=generator).split(BATCH):
            clean = data[batch]
            noisy = clean.clone()
            continuous = [0, 1, 2, 3, 5, 6, 7]
            noisy[:, continuous] += 0.15 * torch.randn(
                (len(batch), len(continuous)), generator=generator
            )
            optimizer.zero_grad()
            (candidate(noisy) - clean).square().mean().backward()
            optimizer.step()
    return candidate.eval()


def experiment(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    known = frames["train"].anomaly.to_numpy()
    replay = arrays["train"][rng.choice(np.flatnonzero(known == 0), REPLAY, replace=False)]
    validation = frames["validation"]
    # Split by customer so gate/calibration and the simulated annotation stream are disjoint.
    customers = np.unique(validation.customer_id)
    query_mask = validation.customer_id.isin(customers[:len(customers) // 2]).to_numpy()
    gate = arrays["validation"][~query_mask]
    gate_y = validation.anomaly.to_numpy()[~query_mask]
    stream = validation.loc[query_mask].sort_values("timestamp_seconds")
    query_x = stream[features].to_numpy()
    query_y = stream.anomaly.to_numpy()
    test = arrays["test"]
    test_y = frames["test"].anomaly.to_numpy()
    # Test panels use the same frozen rows to isolate the effect of the controlled shift.
    strengths = (0.0, 1.0, 1.25, 1.5, 1.75)
    arms = {name: (original, threshold, threshold, "m2") for name in ("active64", "random64", "full")}
    rows = []
    queries = []
    versions = []
    old_floor = evaluate(original, gate, gate_y, threshold)["macro_f1"] - 0.03
    original_rate = (score(original, standardize(gate), True) >= threshold).mean()
    consecutive = 0

    for cycle, (strength, positions) in enumerate(zip(strengths, np.array_split(
            np.arange(len(stream)), len(strengths))), 1):
        pool = shift(query_x[positions], strength)
        labels = query_y[positions]
        drifted_test = shift(test, strength)
        drifted_gate = shift(gate, strength)
        price_psi = psi(arrays["train"][:, PRICE], pool[:, PRICE])
        location_psi = psi(arrays["train"][:, LOCATION], pool[:, LOCATION])
        consecutive = consecutive + 1 if min(price_psi, location_psi) >= 0.25 else 0
        alert = consecutive >= 2
        for arm, (current, cutoff, clean_cutoff, parent) in arms.items():
            pre = evaluate(current, drifted_test, test_y, cutoff)
            selected = np.array([], dtype=int)
            normals_used = 0
            update_ms = query_ms = 0.0
            accepted = False
            candidate_gate_f1 = candidate_old_f1 = candidate_gate_ap = incumbent_gate_ap = None
            before_rss = rss_mib()
            after_rss = before_rss
            if alert:
                start = perf_counter()
                scores = score(current, standardize(pool), True)
                if arm == "active64":
                    selected = active_query(scores, cutoff, BUDGET)
                elif arm == "random64":
                    selected = rng.choice(len(pool), BUDGET, replace=False)
                else:
                    selected = np.arange(len(pool))
                decisions = labels[selected]  # Offline oracle simulates reviewer labels.
                query_ms = (perf_counter() - start) * 1000
                queries.extend({"cycle": cycle, "arm": arm,
                                "transaction_id": stream.iloc[positions[i]].transaction_id,
                                "review_label": int(label)}
                               for i, label in zip(selected, decisions))
                normals = pool[selected[decisions == 0]]
                normals_used = len(normals)
                if normals_used >= 16:
                    start = perf_counter()
                    candidate = update(current, normals, replay, SEED + cycle)
                    candidate_clean_scores = score(candidate, standardize(gate), True)
                    candidate_clean_cutoff = float(np.quantile(candidate_clean_scores, 1 - original_rate))
                    candidate_scores = score(candidate, standardize(drifted_gate), True)
                    candidate_cutoff = float(np.quantile(candidate_scores, 1 - original_rate))
                    candidate_gate = evaluate(candidate, drifted_gate, gate_y, candidate_cutoff)
                    incumbent_gate = evaluate(current, drifted_gate, gate_y, cutoff)
                    candidate_gate_f1 = candidate_gate["macro_f1"]
                    candidate_gate_ap = candidate_gate["average_precision"]
                    incumbent_gate_ap = incumbent_gate["average_precision"]
                    candidate_old_f1 = evaluate(candidate, gate, gate_y, candidate_clean_cutoff)["macro_f1"]
                    accepted = (
                        candidate_gate_ap > incumbent_gate_ap + 0.005
                        and candidate_gate_f1 >= incumbent_gate["macro_f1"] - 0.04
                        and candidate_old_f1 >= old_floor
                    )
                    if accepted:
                        version = f"{arm}-c{cycle}"
                        path = output / f"{version}.pt"
                        torch.save({"state": candidate.state_dict(), "threshold": candidate_cutoff,
                                    "clean_threshold": candidate_clean_cutoff,
                                    "features": features, "scaler_mean": bundle["scaler_mean"],
                                    "scaler_scale": bundle["scaler_scale"],
                                    "bottleneck": bundle["bottleneck"], "version": version,
                                    "parent": parent, "cycle": cycle, "seed": SEED + cycle}, path)
                        versions.append({"version": version, "parent": parent,
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                        current, cutoff, clean_cutoff, parent = (
                            candidate, candidate_cutoff, candidate_clean_cutoff, version
                        )
                        arms[arm] = (current, cutoff, clean_cutoff, parent)
                    update_ms = (perf_counter() - start) * 1000
                    after_rss = rss_mib()
            post = evaluate(current, drifted_test, test_y, cutoff)
            clean = evaluate(current, test, test_y, clean_cutoff)
            rows.append({"cycle": cycle, "strength": strength, "arm": arm,
                         "pool": len(pool), "price_psi": price_psi, "location_psi": location_psi,
                         "alert": alert, "labels": len(selected), "normal_labels": normals_used,
                         "accepted": accepted, "version": parent, "query_ms": query_ms,
                         "update_ms": update_ms, "rss_before_mib": before_rss,
                         "rss_after_mib": after_rss, "candidate_gate_f1": candidate_gate_f1,
                         "candidate_old_f1": candidate_old_f1,
                         "candidate_gate_ap": candidate_gate_ap, "incumbent_gate_ap": incumbent_gate_ap,
                         **{f"pre_{k}": v for k, v in pre.items()},
                         **{f"post_{k}": v for k, v in post.items()},
                         **{f"clean_{k}": v for k, v in clean.items()}})

    result = pd.DataFrame(rows)
    result.to_csv(output / "cycles.csv", index=False)
    pd.DataFrame(queries).to_csv(output / "review_audit.csv", index=False)
    if versions:
        latest = versions[-1]
        path = output / f"{latest['version']}.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != latest["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for {path.name}")
        saved = torch.load(path, weights_only=True, map_location="cpu")
        restored = autoencoder(len(features), saved["bottleneck"])
        restored.load_state_dict(saved["state"])
        live_model = next(arm[0] for arm in arms.values() if arm[-1] == latest["version"])
        np.testing.assert_array_equal(
            score(restored.eval(), standardize(test[:32]), True),
            score(live_model, standardize(test[:32]), True),
        )
    noise = test.copy()
    indices = [0, 1, 2, 3, 5, 6, 7]
    noise[:, indices] += rng.normal(0, .25, (len(test), len(indices))) * np.asarray(
        bundle["scaler_scale"]
    )[indices]
    final = {}
    for name, (model, cutoff, clean_cutoff, version) in {"frozen": (original, threshold, threshold, "m2"), **arms}.items():
        final[name] = {"version": version,
                       "clean": evaluate(model, test, test_y, clean_cutoff),
                       "perturbed": evaluate(model, noise, test_y, clean_cutoff),
                       "drift": evaluate(model, shift(test, strengths[-1]), test_y, cutoff),
                       "efficiency": benchmark(model, test),
                       "checkpoint_bytes": (output / f"{version}.pt").stat().st_size
                       if version != "m2" else checkpoint.stat().st_size}
    (output / "summary.json").write_text(json.dumps({
        "seed": SEED, "hardware": hardware(), "versions": versions, "final": final,
        "baseline_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "baseline_bytes": checkpoint.stat().st_size,
        "gate_clean_macro_f1_floor": old_floor,
    }, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for name, group in result.groupby("arm"):
        axes[0].plot(group.cycle, group.post_macro_f1, "o-", label=name)
        axes[1].plot(group.cycle, group.post_average_precision, "o-", label=name)
        axes[2].plot(group.cycle, group.labels.cumsum(), "o-")
    active = result[result.arm == "active64"]
    axes[0].plot(active.cycle, active.pre_macro_f1, "k--", label="active64 before update")
    full = result[result.arm == "full"]
    axes[1].plot(full.cycle, full.pre_average_precision, "k--", label="full before update")
    axes[1].legend(fontsize=7)
    for ax, title in zip(axes, ("Drifted macro F1", "Drifted average precision", "Cumulative reviews")):
        ax.set(xlabel="Cycle", title=title, xticks=active.cycle)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    experiment(parser.parse_args().output)
