"""Run the Milestone 3 transaction scoring and monitoring experiments on CPU."""
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "milestone2"
OUT = ROOT / "results"
sys.path.insert(0, str(BASE))
from models import autoencoder, hardware, rss_mib, score

SEED = 6450
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
OUT.mkdir(exist_ok=True)
frames = {name: pd.read_csv(BASE / f"data/features_{name}.csv")
          for name in ("train", "validation", "test")}
features = frames["train"].columns[6:].tolist()
checkpoint = BASE / "models/detectors.pt"
bundle = torch.load(checkpoint, weights_only=True, map_location="cpu")
mean = np.asarray(bundle["scaler_mean"])
scale = np.asarray(bundle["scaler_scale"])
primary = bundle["primary"]
original = autoencoder(len(features), bundle["bottleneck"])
original.load_state_dict(bundle["states"][primary])
original.eval()
threshold = bundle["thresholds"][primary]
arrays = {name: frame[features].to_numpy() for name, frame in frames.items()}

def standardize(values):
    return ((values - mean) / scale).astype("float32")

def sanitized(values):
    """Impute unavailable features at the training median for offline evaluation."""
    median = np.median(arrays["train"], axis=0)
    return np.where(np.isfinite(values), values, median)

def shifted(values):
    shifted_values = values.copy()
    shifted_values[:, 6] += 0.65
    shifted_values[:, 5] += 0.45
    return shifted_values

def perturb(values, strength):
    """Fixed, model-independent sensitivity test on synthetic transaction features."""
    changed = values.copy()
    changed[:, 0] += strength * scale[0]
    changed[:, 1] += strength * scale[1]
    return changed

def evaluate(model, values, labels, cutoff):
    scores = score(model, standardize(sanitized(values)), True)
    flags = scores >= cutoff
    report = classification_report(labels, flags, output_dict=True, zero_division=0)
    return {
        "precision": report["1"]["precision"], "recall": report["1"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "auroc": roc_auc_score(labels, scores), "average_precision": average_precision_score(labels, scores),
        "false_alarms": int((flags & (labels == 0)).sum()),
        "misses": int((~flags & (labels == 1)).sum()),
        "n": len(labels), "positive": int(labels.sum()),
    }

def benchmark(model, values):
    for row in values[:40]:
        score(model, standardize(sanitized(row[None, :])), True)
    samples = []
    median = np.median(arrays["train"], axis=0)
    rss_before = rss_mib()
    started = perf_counter()
    for row in values[:1000]:
        start = perf_counter()
        prepared = np.where(np.isfinite(row), row, median)
        score(model, standardize(prepared[None, :]), True)
        samples.append((perf_counter() - start) * 1000)
    elapsed = perf_counter() - started
    return {"p50_ms": float(np.percentile(samples, 50)), "p90_ms": float(np.percentile(samples, 90)),
            "rows_per_second": len(samples) / elapsed, "process_rss_mib": rss_mib(),
            "rss_before_mib": rss_before,
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters())}

def main():
    adapted = copy.deepcopy(original)
    optimizer = torch.optim.Adam(adapted.parameters(), lr=0.001, weight_decay=0.0001)
    adapt_pool = np.concatenate((arrays["train"], shifted(arrays["train"])))
    target = torch.from_numpy(standardize(adapt_pool))
    started = perf_counter()
    training_rss = rss_mib()
    adapted.train()
    for epoch in range(15):
        for indices in torch.randperm(len(target)).split(256):
            clean = target[indices]
            noisy = clean.clone()
            continuous = [0, 1, 2, 3, 5, 6, 7]
            noisy[:, continuous] += 0.15 * torch.randn_like(clean[:, continuous])
            noisy[torch.rand_like(clean) < 0.06] = 0
            optimizer.zero_grad()
            loss = (adapted(noisy) - clean).square().mean()
            loss.backward()
            optimizer.step()
    training_seconds = perf_counter() - started
    adapted.eval()
    validation = arrays["validation"]
    validation_shift = shifted(validation)
    validation_labels = frames["validation"].anomaly.to_numpy()
    original_val = score(original, standardize(validation), True)
    flag_rate = float(np.mean(original_val >= threshold))
    adapted_val = score(adapted, standardize(validation_shift), True)
    adapted_threshold = float(np.quantile(adapted_val, 1 - flag_rate))
    calibrators = {
        "original": IsotonicRegression(out_of_bounds="clip").fit(original_val, validation_labels),
        "adapted": IsotonicRegression(out_of_bounds="clip").fit(adapted_val, validation_labels),
    }
    torch.save({"state": adapted.state_dict(), "threshold": adapted_threshold,
                "features": features, "seed": SEED, "scaler_mean": mean.tolist(),
                "scaler_scale": scale.tolist(),
                "calibration_x": calibrators["adapted"].X_thresholds_.tolist(),
                "calibration_y": calibrators["adapted"].y_thresholds_.tolist()}, OUT / "adapted.pt")

    test = arrays["test"]
    labels = frames["test"].anomaly.to_numpy()
    noise = test.copy()
    noise[:, continuous] += rng.normal(0, 0.25, (len(test), len(continuous))) * scale[continuous]
    loss_input = test.copy()
    loss_input[rng.random(len(test)) < 0.35, 7] = np.nan
    ood = test.copy()
    ood[:, 5] += 2.0
    ood[:, 6] += 1.5
    rare = (labels == 0) | (rng.random(len(test)) < 0.20)
    scenarios = {"clean": (test, labels), "noise": (noise, labels), "partial_loss": (loss_input, labels),
                 "ood": (ood, labels), "rare": (test[rare], labels[rare]), "drift": (shifted(test), labels)}
    for strength in (0.25, 0.5, 1.0, 2.0):
        scenarios[f"bounded_{strength}"] = (perturb(test, strength), labels)
    variants = {"original": (original, threshold), "adapted": (adapted, adapted_threshold)}
    metrics = []
    timings = {}
    for name, (model, cutoff) in variants.items():
        timings[name] = {}
        for scenario, (values, truth) in scenarios.items():
            metrics.append({"model": name, "scenario": scenario, **evaluate(model, values, truth, cutoff)})
            if scenario in ("clean", "noise", "partial_loss", "bounded_2.0", "drift"):
                timings[name][scenario] = benchmark(model, values)
    results = pd.DataFrame(metrics)
    results.to_csv(OUT / "metrics.csv", index=False)

    drift_values = scenarios["drift"][0]
    raw_scores = {name: score(model, standardize(drift_values), True)
                  for name, (model, cutoff) in variants.items()}
    flags = {name: raw_scores[name] >= cutoff for name, (model, cutoff) in variants.items()}
    probs = {name: calibrators[name].predict(raw_scores[name]) for name in variants}
    old_errors = flags["original"] != labels
    resolved = old_errors & (flags["adapted"] == labels)
    remaining = old_errors & ~resolved
    selected = np.r_[np.flatnonzero(resolved & (labels == 1))[:2],
                     np.flatnonzero(resolved & (labels == 0))[:2],
                     np.flatnonzero(remaining & (labels == 1))[:2],
                     np.flatnonzero(remaining & (labels == 0))[:2]]
    if len(selected) < 8:
        selected = np.r_[selected, np.setdiff1d(np.flatnonzero(old_errors), selected)[:8-len(selected)]]
    rows = []
    for i in selected:
        transaction = frames["test"].iloc[i]
        rows.append({"id": transaction.transaction_id, "scenario": transaction.scenario,
                     "expected": int(labels[i]), "original": int(flags["original"][i]),
                     "adapted": int(flags["adapted"][i]), "resolved": bool(resolved[i]),
                     "original_probability": float(probs["original"][i]),
                     "adapted_probability": float(probs["adapted"][i]),
                     "log_unit_price": float(drift_values[i, 6]),
                     "log_location_shift": float(drift_values[i, 5]),
                     "amount_deviation": float(drift_values[i, 0])})
    pd.DataFrame(rows).to_csv(OUT / "failures.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    bounded = ["clean", "bounded_0.25", "bounded_0.5", "bounded_1.0", "bounded_2.0"]
    for name in variants:
        subset = results[results.model == name].set_index("scenario")
        axes[0].plot([0, .25, .5, 1, 2], subset.loc[bounded, "macro_f1"], "o-", label=name)
        axes[1].hist(probs[name], bins=np.linspace(0, 1, 11), alpha=.5, label=name)
        bin_ids = np.minimum((probs[name] * 10).astype(int), 9)
        pairs = [(probs[name][bin_ids == b].mean(), labels[bin_ids == b].mean())
                 for b in np.unique(bin_ids)]
        axes[2].plot(*np.asarray(pairs).T, "o-", label=name)
    axes[0].set(xlabel="Perturbation (training SD)", ylabel="Macro F1", title="Sensitivity")
    axes[1].set(xlabel="Estimated anomaly probability", ylabel="Count", title="Shifted test confidence")
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].set(xlabel="Mean estimated probability", ylabel="Observed anomaly fraction", title="Shifted test reliability")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png", dpi=160)
    plt.close(fig)

    groups = frames["test"].customer_id.to_numpy()
    group_rows = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    intervals = {}
    for name in variants:
        samples = []
        for _ in range(200):
            idx = np.concatenate([group_rows[i] for i in rng.integers(len(group_rows), size=len(group_rows))])
            samples.append(average_precision_score(labels[idx], raw_scores[name][idx]))
        intervals[name] = np.percentile(samples, [2.5, 97.5]).tolist()

    metadata = {"hardware": hardware(), "platform": platform.platform(), "seed": SEED,
                "adaptation_seconds": training_seconds, "training_rss_before_mib": training_rss,
                "training_rss_after_mib": rss_mib(), "original_threshold": threshold,
                "adapted_threshold": adapted_threshold, "validation_flag_rate": flag_rate,
                "timings": timings, "shifted_ap_95_customer_bootstrap": intervals,
                "resolved_count": int(resolved.sum()), "features": features,
                "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [checkpoint, *(BASE / f"data/features_{n}.csv" for n in frames)]}}
    (OUT / "experiment.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
