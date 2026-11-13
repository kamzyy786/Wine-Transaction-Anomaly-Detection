# Milestone 3 | Wine Transaction Anomaly Detection

- `run.py` generates robustness results and the adapted checkpoint.
- `monitor.py` produces the monitoring dashboard and simulated audit log.
- `service.py` exposes the hybrid scoring path over loopback HTTP with bounded admission and model switching.
- `load_test.py` measures request latency and tests rollback and overload recovery.
- `guard.py` defines the offline reference review policy.

From the repository root, with the dependencies in `milestone3/requirements.txt`:

```sh
python milestone3/run.py
python milestone3/monitor.py
python milestone3/load_test.py
```

The service uses the adjacent Milestone 2 assets. Its state is in memory, and
the model switch endpoint is intended for the local test environment.
