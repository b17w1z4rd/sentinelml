# SentinelML

An endpoint telemetry threat-scoring **research baseline** with synthetic data, episode-disjoint evaluation, a validation-selected alert threshold, and a reproducible CLI. Designed as a starting point for ransomware and C2 detection experiments. Python 3.10+.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
sentinelml generate --out data/synthetic.csv --episodes 600
sentinelml train --data data/synthetic.csv --model artifacts/model.joblib --report artifacts/report.json
sentinelml score --model artifacts/model.joblib --input examples/record.json
python -m unittest discover -s tests -v
```

`generate` creates correlated telemetry windows within each simulated episode. `train` makes a 60/20/20 train/validation/test split **by episode_id**, keeping all windows from an episode in the same split. It trains a histogram gradient boosting classifier, chooses the lowest useful threshold satisfying a minimum validation precision (default 0.8), then evaluates once on the held-out test set. The JSON report includes average precision, ROC AUC, Brier score, precision, recall, F1, and confusion matrix `[TN, FP, FN, TP]`. Model files and synthetic CSVs are excluded from Git.

## Use your own telemetry

Provide a CSV with these columns:

| Column | Meaning |
| --- | --- |
| `episode_id` | Independent host or incident episode identifier; do not reuse across unrelated cases |
| `label` | 0 benign, 1 threat; one consistent label per episode |
| `file_entropy` | Approximate byte entropy, 0–8 bits/byte |
| `file_write_rate`, `rename_rate`, `process_burst`, `dns_query_rate` | Counts per fixed collection window; use the same window length for training and inference |
| `rare_domain_ratio` | Share of queries to uncommon domains, 0–1 |
| `shadow_delete`, `unsigned_process` | Boolean indicators, 0 or 1 |

```bash
sentinelml train --data /path/to/labeled_windows.csv --min-precision 0.9
```

A production study should also hold out hosts and time periods, deduplicate campaigns, audit label quality, measure false alerts per host day, and compare against simple rules. The included generator makes patterns that the model can learn; synthetic test scores **do not establish real-world detection performance**. Alert scores are estimates from the training distribution, not calibrated operational risk. Never load an untrusted `.joblib` model: it can execute code during deserialization.

## Architecture

```text
CSV telemetry → schema checks → episode-disjoint train/validation/test
                              → HistGradientBoostingClassifier
validation labels + scores → precision-constrained threshold
held-out test → metrics JSON
JSON record + saved model → score and alert decision
```

## Next experiments

- Ingest anonymized Sysmon/Wazuh events, derive fixed time-window features, and retain source-event provenance.
- Compare with an isolation forest, a rules baseline, and a time-held-out evaluation.
- Add probability calibration on a separate calibration split and drift monitoring over live data.

MIT License. See [LICENSE](LICENSE).
