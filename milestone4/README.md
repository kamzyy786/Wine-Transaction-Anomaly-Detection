# Milestone 4 | Continual learning and simulated human review

From the repository root with Python 3.12:

```sh
python -m pip install -r milestone4/requirements.txt
python milestone4/run.py --output milestone4/reproduced-results
python milestone4/latency.py --output milestone4/reproduced-results/inference_overlap.json
```

The script loads the checked-in Milestone 2 PyTorch autoencoder and derived
feature splits, scores a stream with simulated price and location drift, uses
the Milestone 3 PSI alert, simulates human review with the dataset's existing
synthetic anomaly labels, and evaluates replay-trained candidates. It verifies
that a saved, accepted checkpoint reloads with identical scores on 32 rows.
No raw transaction data or network service is needed to run this demo.

The active arm queries 32 near-threshold records and 32 likely normal records
per alert. The random arm queries 64 random records. Full review queries all
375 records per alert. The gate compares drifted validation average precision
and macro F1 and checks macro F1 on the clean validation panel. The final test
panels stay outside the gate. Training and selection never consult test labels.

The versioned reference run is under `milestone4/results/`. Outputs are `cycles.csv` (PSI, review counts, before/after metrics and timing),
`review_audit.csv` (simulated labels), `summary.json` (final results and version
hashes), `trajectories.png`, accepted `.pt` checkpoints, and `inference_overlap.json` (model-only latency with and without a concurrent replay update). Use a fresh output
directory for each run to keep separate run histories. Runtime and RSS vary by
host. The review and promotion checks run offline. The Milestone 3 service separately demonstrates request handling, overload recovery, and rollback.

The experiment seed is 64504. The reference run uses Python 3.12.14 and PyTorch
2.15.0+cpu on one PyTorch CPU thread. The data and initial checkpoints are
versioned in the repository, and the output manifest records the baseline
SHA-256. The overlap benchmark times the incumbent on pre-scaled inputs
while a separate thread trains a copy of the model for three trials.

Earlier stages: [Milestone 2](../milestone2/README.md) and
[Milestone 3](../milestone3/README.md). For the root preprocessing stage,
install the root requirements and run `python process.py`.

## Pipeline entry points

| Stage | Entry point | Outputs |
| --- | --- | --- |
| Data preparation | `python process.py` | `outputs/` |
| History generation and model training | `cd milestone2 && python experiment.py` | `milestone2/data/`, `milestone2/models/`, `milestone2/results/` |
| Robustness tests and adaptation | `python milestone3/run.py` | `milestone3/results/metrics.csv`, `adapted.pt` |
| Monitoring and deployment validation | `python milestone3/monitor.py` and `python milestone3/load_test.py` after the robustness run | `milestone3/results/monitoring.csv`, `deployment.json` |
| Continual learning and human review | `python milestone4/run.py --output milestone4/reproduced-results` | Per-cycle metrics, trajectories, review audit, and versioned checkpoints |
