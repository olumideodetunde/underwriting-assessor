"""Gate metrics smoke check for the frequency model.

Retrains the claims-frequency model on the committed dataset (via
``config/smoke.yaml``) with no MLflow tracking and no S3, prints the
train/test metrics table, and saves the four diagnostic plots under
``.no-mistakes/evidence/``. The stdout table and saved figures become the
data-science evidence attached to the gated PR.

It deliberately reuses ``train_and_evaluate()`` from ``src.train.frequency``
(the tracking-free half of ``run()``) so the numbers match the real training
path without needing a tracking server or AWS credentials.

Run from the repo root:

    python -m scripts.gate_metrics_smoke
"""

from __future__ import annotations

import matplotlib

# Headless backend: the gate has no display and we only save figures to disk.
matplotlib.use("Agg")

from pathlib import Path

from src.config import load_config
from src.train.frequency import train_and_evaluate

CONFIG_PATH = "config/smoke.yaml"
EVIDENCE_DIR = Path(".no-mistakes/evidence")

# Order + labels for the printed metrics table. Keys match calculate_metrics().
METRIC_LABELS = [
    ("mse", "MSE"),
    ("rmse", "RMSE"),
    ("mae", "MAE"),
    ("r2", "R2"),
    ("medae", "MedAE"),
    ("poisson_deviance", "Poisson deviance"),
]


def _print_metrics_table(train_metrics: dict, test_metrics: dict) -> None:
    header = f"{'Metric':<18}{'Train':>16}{'Test':>16}"
    print(header)
    print("-" * len(header))
    for key, label in METRIC_LABELS:
        print(f"{label:<18}{train_metrics[key]:>16.6f}{test_metrics[key]:>16.6f}")


def _save_figures(figures: dict) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, figure in figures.items():
        out_path = EVIDENCE_DIR / filename
        figure.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")


def main() -> None:
    config = load_config(CONFIG_PATH)
    result = train_and_evaluate(config)

    print(f"\nFrequency model metrics ({config['frequency']['algorithm']})\n")
    _print_metrics_table(result["train_metrics"], result["test_metrics"])
    print()
    _save_figures(result["figures"])


if __name__ == "__main__":
    main()
