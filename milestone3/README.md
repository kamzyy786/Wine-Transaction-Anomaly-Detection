# Milestone 3 | Wine Transaction Anomaly Detection

Run from the repository root with Python 3.12 and the dependencies in
`milestone3/requirements.txt`. A CPU build of PyTorch is sufficient.

```sh
python milestone3/run.py
python milestone3/monitor.py
```

The first command loads the frozen Milestone 2 checkpoint and feature CSVs,
trains a denoising adaptation, and writes metrics, timings, plots, failure
cases, and the adapted checkpoint. The second writes the simulated monitoring
dashboard and audit CSV. Keep the adjacent
`milestone2` directory in place. The feature perturbations and monitoring
windows are synthetic. Timing and process memory depend on the host.
`guard.py` provides the reference feature validation and review policy.
