"""Endpoint telemetry experiment. Data generated here is synthetic, not evidence of field performance."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

FEATURES = ("file_entropy", "file_write_rate", "rename_rate", "process_burst",
            "dns_query_rate", "rare_domain_ratio", "shadow_delete", "unsigned_process")
REQUIRED = ("episode_id", "label", *FEATURES)


def generate(path, episodes=600, seed=42):
    """Make correlated windows grouped by simulated host episode."""
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("episode_id", "label", *FEATURES))
        writer.writeheader()
        for episode in range(episodes):
            threat = int(rng.random() < .23)
            host_bias = rng.normal(0, .65)
            subtype = rng.choice(["encrypt", "c2", "mixed"]) if threat else "benign"
            for _ in range(int(rng.integers(3, 9))):
                signal = rng.normal(0, 1)
                encrypt = int(subtype in ("encrypt", "mixed"))
                c2 = int(subtype in ("c2", "mixed"))
                vals = [
                    np.clip(5.4 + .9*encrypt + .35*signal + .28*host_bias + rng.normal(0,.6),0,8),
                    max(0, 9 + 18*encrypt + 4*signal + 3*host_bias + rng.normal(0,10)),
                    max(0, 1 + 12*encrypt + 2*signal + rng.normal(0,6)),
                    max(0, 3 + 5*threat + host_bias + rng.normal(0,3)),
                    max(0, 5 + 19*c2 + 3*signal + rng.normal(0,12)),
                    np.clip(.12 + .46*c2 + .06*signal + rng.normal(0,.17),0,1),
                    int(rng.random() < (.24 if encrypt else .008)),
                    int(rng.random() < (.47 if threat else .11)),
                ]
                writer.writerow(dict(zip(REQUIRED, (f"episode-{episode:05d}", threat,
                                                       *(round(float(v), 5) for v in vals)))))
    return path


def load_csv(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty dataset")
    groups = np.array([r["episode_id"] for r in rows])
    try:
        labels = np.array([int(r["label"]) for r in rows])
        x = np.array([[float(r[k]) for k in FEATURES] for r in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid numeric value") from exc
    if not np.isfinite(x).all() or not set(labels).issubset({0, 1}):
        raise ValueError("Features must be finite and labels must be 0 or 1")
    by_group = {}
    for group, label in zip(groups, labels):
        if group in by_group and by_group[group] != label:
            raise ValueError("An episode_id has conflicting labels")
        by_group[group] = label
    return x, labels, groups


def split_groups(x, y, groups, seed=42):
    outer = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    dev, test = next(outer.split(x, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed+1)
    tr_local, va_local = next(inner.split(x[dev], y[dev], groups[dev]))
    train, val = dev[tr_local], dev[va_local]
    if any(len(set(y[idx])) != 2 for idx in (train, val, test)):
        raise ValueError("Each split needs both classes; add more episodes")
    return train, val, test


def choose_threshold(y, scores, min_precision=.8):
    # Threshold is selected using validation data only, never test labels.
    candidates = np.unique(np.r_[0., scores, 1.])
    choices = []
    for threshold in candidates:
        pred = scores >= threshold
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        fn = np.sum(~pred & (y == 1))
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        if precision >= min_precision:
            choices.append((recall, precision, float(threshold)))
    if not choices:
        raise ValueError("No threshold meets minimum validation precision")
    return max(choices, key=lambda item: (item[0], item[1], item[2]))[2]


def metrics(y, scores, threshold):
    pred = scores >= threshold
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "brier_score": float(brier_score_loss(y, scores)),
            "precision": float(p), "recall": float(r), "f1": float(f),
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, pred, labels=[0,1]).ravel().tolist()}


def train(data_path, model_path, report_path, seed=42, min_precision=.8):
    x, y, groups = load_csv(data_path)
    train_idx, val_idx, test_idx = split_groups(x, y, groups, seed)
    model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=12,
                                            l2_regularization=2., random_state=seed)
    model.fit(x[train_idx], y[train_idx])
    val_scores = model.predict_proba(x[val_idx])[:, 1]
    threshold = choose_threshold(y[val_idx], val_scores, min_precision)
    test_scores = model.predict_proba(x[test_idx])[:, 1]
    report = {"dataset": str(data_path), "synthetic_data_warning": "Demo data only; no real-world performance claim",
              "seed": seed, "minimum_validation_precision": min_precision,
              "rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
              "episodes": {"train": len(set(groups[train_idx])), "validation": len(set(groups[val_idx])),
                           "test": len(set(groups[test_idx]))},
              "threshold": threshold, "validation": metrics(y[val_idx], val_scores, threshold),
              "test": metrics(y[test_idx], test_scores, threshold)}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold}, model_path)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


def score(model_path, record):
    # Only load model files you created or trust: joblib uses pickle.
    bundle = joblib.load(model_path)
    features = bundle["features"]
    try:
        values = np.array([[float(record[k]) for k in features]])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric features: {', '.join(features)}") from exc
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    probability = float(bundle["model"].predict_proba(values)[0, 1])
    return {"threat_probability": probability, "alert": probability >= bundle["threshold"],
            "threshold": bundle["threshold"]}
