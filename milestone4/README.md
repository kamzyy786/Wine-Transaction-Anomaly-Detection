# Milestone 4 | Continual learning and simulated human review

From the repository root with Python 3.12:

```sh
python -m pip install -r milestone3/requirements.txt
python milestone4/run.py --output milestone4/results
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

Outputs are `cycles.csv` (PSI, review counts, before/after metrics and timing),
`review_audit.csv` (simulated labels), `summary.json` (final results and version
hashes), `trajectories.png`, and accepted `.pt` checkpoints. Use a fresh output
directory for each run to keep separate run histories. Runtime and RSS vary by
host. The review queue, promotion checks and rollback are simulated offline;
the existing Milestone 3 service is a separate local deployment demonstration.

Earlier stages: [Milestone 2](../milestone2/README.md) and
[Milestone 3](../milestone3/README.md). For the root preprocessing stage,
install the root requirements and run `python process.py`.
