"""Simulate chronological monitoring windows and write a dashboard and audit CSV."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from scipy.stats import wasserstein_distance

from run import BASE, OUT, arrays, features, frames, original, sanitized, score, shifted, standardize, threshold


def psi(reference, current):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    bins = np.r_[-np.inf, edges[1:-1], np.inf]
    first = np.histogram(reference, bins)[0] + 0.5
    second = np.histogram(current, bins)[0] + 0.5
    p, q = first / first.sum(), second / second.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def main():
    positions = np.argsort(frames["test"].timestamp_seconds.to_numpy())
    truth = frames["test"].anomaly.to_numpy()
    rows = []
    consecutive = 0
    for window, indices in enumerate(np.array_split(positions, 10), 1):
        values = arrays["test"][indices].copy()
        if window >= 5:
            values = shifted(values)
        if window >= 8:
            values[::5, 7] = np.nan
        prepared = sanitized(values)
        scores = score(original, standardize(prepared), True)
        flags = scores >= threshold
        price_psi = psi(arrays["train"][:, 6], values[:, 6])
        location_psi = psi(arrays["train"][:, 5], values[:, 5])
        consecutive = consecutive + 1 if min(price_psi, location_psi) >= .25 else 0
        rows.append({
            "window": window, "rows": len(indices), "unit_price_psi": price_psi,
            "location_psi": location_psi,
            "price_wasserstein": wasserstein_distance(arrays["train"][:, 6], values[:, 6]),
            "chemistry_null_rate": float(np.isnan(values[:, 7]).mean()),
            "flag_rate": float(flags.mean()), "drift_alert": consecutive >= 2,
            "quality_alert": float(np.isnan(values[:, 7]).mean()) > .05,
            "macro_f1_on_label_arrival": f1_score(truth[indices], flags, average="macro"),
            "ap_on_label_arrival": average_precision_score(truth[indices], scores),
            "label_available_window": window + 2,
        })
    data = pd.DataFrame(rows)
    data.to_csv(OUT / "monitoring.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2))
    axes[0, 0].plot(data.window, data.unit_price_psi, "o-", label="Unit price")
    axes[0, 0].plot(data.window, data.location_psi, "s-", label="Location")
    axes[0, 0].axhline(.25, color="gray", ls="--")
    axes[0, 0].legend()
    axes[0, 0].set(title="Population stability index")
    axes[0, 1].plot(data.window, data.chemistry_null_rate, "o-")
    axes[0, 1].axhline(.05, color="gray", ls="--")
    axes[0, 1].set(title="Chemistry missing fraction")
    axes[1, 0].plot(data.window, data.flag_rate, "o-")
    axes[1, 0].set(title="Review flag fraction")
    axes[1, 1].plot(data.label_available_window, data.macro_f1_on_label_arrival, "o-", label="Macro F1")
    axes[1, 1].plot(data.label_available_window, data.ap_on_label_arrival, "s-", label="AP")
    axes[1, 1].legend()
    axes[1, 1].set(title="Quality after simulated label delay")
    for ax in axes.flat:
        ax.set_xlabel("Window")
    fig.tight_layout()
    fig.savefig(OUT / "dashboard.png", dpi=160)


if __name__ == "__main__":
    main()
