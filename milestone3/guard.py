"""Reference policy for a single derived-feature record before automatic disposition."""
import numpy as np

from run import arrays, features, sanitized


def prepare(record):
    if set(record) != set(features):
        raise ValueError("Feature schema mismatch")
    values = np.asarray([record[name] if record[name] is not None else np.nan
                         for name in features], dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite value")
    missing = np.isnan(values)
    for i in (2, 3, 5, 6):
        if values[i] < 0:
            raise ValueError(f"Negative log feature: {features[i]}")
    for i in (4, 8):
        if not missing[i] and values[i] not in (0, 1):
            raise ValueError(f"Invalid binary feature: {features[i]}")
    if not missing[7] and not 0 <= values[7] <= 1:
        raise ValueError("Chemical disagreement outside [0, 1]")
    return sanitized(values), bool(missing.any())


def disposition(score, threshold, probability, incomplete):
    if incomplete or not np.isfinite(score) or not 0 <= probability <= 1:
        return "manual_review"
    if score >= threshold or 0.40 <= probability <= 0.60:
        return "manual_review"
    return "no_model_flag"
