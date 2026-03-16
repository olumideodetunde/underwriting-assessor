"""
Lightweight end-to-end test for src.train.frequency.run().

Strategy
--------
- Build a small synthetic DataFrame that mirrors the real CSV columns
  (including the Date_* columns that Driver needs and the Type_fuel /
  Value_vehicle columns that Vehicle needs).
- Patch only two boundaries:
    1. load_csv  → return the synthetic DataFrame (no file I/O)
    2. tracking  → mock every MLflow call (no running server)
- Everything else (split, feature engineering, factory, model fitting,
  metrics, plots) runs for real so we know the pipeline actually hangs
  together.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from matplotlib.figure import Figure

from src.train.frequency import run
from matplotlib import pyplot as plt


# ── helpers ───────────────────────────────────────────

def _make_synthetic_dataset(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a small DataFrame with every column the pipeline touches."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Year_matriculation":    rng.integers(2000, 2023, size=n),
        "Type_risk":             rng.integers(1, 5, size=n),
        "Area":                  rng.integers(1, 7, size=n),
        "Value_vehicle":         rng.uniform(5_000, 50_000, size=n).round(2),
        "Distribution_channel":  rng.integers(1, 4, size=n),
        "Cylinder_capacity":     rng.integers(800, 3000, size=n),
        "N_claims_year":         rng.integers(0, 4, size=n),
        "Type_fuel":             rng.choice(["Gasoline", "Diesel", "Hybrid"], size=n),
        "Date_birth":            pd.to_datetime(rng.integers(1960, 2000, size=n), format="%Y").strftime("%d/%m/%Y"),
        "Date_driving_licence":  pd.to_datetime(rng.integers(1980, 2020, size=n), format="%Y").strftime("%d/%m/%Y"),
        "Date_last_renewal":     pd.to_datetime(rng.integers(2020, 2025, size=n), format="%Y").strftime("%d/%m/%Y"),
        "Date_start_contract":   pd.to_datetime(rng.integers(2015, 2023, size=n), format="%Y").strftime("%d/%m/%Y"),
        "Date_next_renewal":     pd.to_datetime(rng.integers(2025, 2027, size=n), format="%Y").strftime("%d/%m/%Y"),
        "Date_lapse":            pd.to_datetime(rng.integers(2025, 2027, size=n), format="%Y").strftime("%d/%m/%Y"),
    })


def _make_config() -> dict:
    return {
        "insurance_csv": "fake/path.csv",
        "features": [
            "Year_matriculation", "Type_risk", "Area",
            "Value_vehicle", "Distribution_channel", "Cylinder_capacity",
        ],
        "target": "N_claims_year",
        "frequency": {
            "algorithm": "poisson_regressor",
            "parameters": {"alpha": 1.0, "max_iter": 1000},
        },
        "tracking": {
            "uri": "http://fake:5000",
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "run_description": "automated test",
            "artifact_model_name": "test_model",
        },
    }


# ── fixtures ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to free memory."""
    yield
    plt.close("all")


@pytest.fixture
def synthetic_data():
    return _make_synthetic_dataset()


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def mock_tracking():
    """Patch every function in the tracking module so no MLflow server is needed."""
    with (
        patch("src.train.frequency.tracking.init") as mock_init,
        patch("src.train.frequency.tracking.start_run") as mock_start_run,
        patch("src.train.frequency.tracking.log_parameters") as mock_log_params,
        patch("src.train.frequency.tracking.log_metrics_nested") as mock_log_metrics,
        patch("src.train.frequency.tracking.log_model") as mock_log_model,
        patch("src.train.frequency.tracking.log_figures") as mock_log_figures,
    ):
        mock_start_run.return_value.__enter__ = MagicMock(return_value="fake-run-id")
        mock_start_run.return_value.__exit__ = MagicMock(return_value=False)
        yield {
            "init": mock_init,
            "start_run": mock_start_run,
            "log_parameters": mock_log_params,
            "log_metrics_nested": mock_log_metrics,
            "log_model": mock_log_model,
            "log_figures": mock_log_figures,
        }


# ── tests ─────────────────────────────────────────────

class TestFrequencyRunEndToEnd:
    """run() completes without error and calls every tracking function."""

    def test_run_completes_without_error(self, synthetic_data, config, mock_tracking):
        with patch("src.train.frequency.load_csv", return_value=synthetic_data):
            run(config)

    def test_tracking_init_called_with_config(self, synthetic_data, config, mock_tracking):
        with patch("src.train.frequency.load_csv", return_value=synthetic_data):
            run(config)

        mock_tracking["init"].assert_called_once_with(
            config["tracking"]["uri"],
            config["tracking"]["experiment_name"],
        )

    def test_start_run_called_with_config(self, synthetic_data, config, mock_tracking):
        with patch("src.train.frequency.load_csv", return_value=synthetic_data):
            run(config)

        mock_tracking["start_run"].assert_called_once_with(
            run_name=config["tracking"]["run_name"],
            description=config["tracking"]["run_description"],
        )

    def test_parameters_logged(self, synthetic_data, config, mock_tracking):
        with patch("src.train.frequency.load_csv", return_value=synthetic_data):
            run(config)

        mock_tracking["log_parameters"].assert_called_once_with(
            config["frequency"]["parameters"],
        )

    def test_train_and_test_metrics_logged(self, synthetic_data, config, mock_tracking):
        with patch("src.train.frequency.load_csv", return_value=synthetic_data):
            run(config)

        calls = mock_tracking["log_metrics_nested"].call_args_list
        assert len(calls) == 2

        train_metrics = calls[0][0][0]
        assert calls[0][1]["prefix"] == "train"
        for key in ("mse", "rmse", "mae", "r2", "medae", "poisson_deviance"):
            assert key in train_metrics

        test_metrics = calls[1][0][0]
        assert calls[1][1]["prefix"] == "test"
        for key in ("mse", "rmse", "mae", "r2", "medae", "poisson_deviance"):
            assert key in test_metrics

    def test_model_logged_with_artifact_name(self, synthetic_data, config, mock_tracking):
        with patch("src.train.frequency.load_csv", return_value=synthetic_data):
            run(config)

        mock_tracking["log_model"].assert_called_once()
        _, kwargs = mock_tracking["log_model"].call_args
        assert kwargs["name"] == config["tracking"]["artifact_model_name"]

    def test_all_four_figures_logged(self, synthetic_data, config, mock_tracking):
        with patch("src.train.frequency.load_csv", return_value=synthetic_data):
            run(config)

        mock_tracking["log_figures"].assert_called_once()
        figures_dict = mock_tracking["log_figures"].call_args[0][0]
        expected_keys = {
            "claims_distribution.png",
            "feature_importance.png",
            "residuals.png",
            "actual_vs_predicted.png",
        }
        assert set(figures_dict.keys()) == expected_keys
        for fig in figures_dict.values():
            assert isinstance(fig, Figure)

    def test_run_returns_nothing(self, synthetic_data, config, mock_tracking):
        with patch("src.train.frequency.load_csv", return_value=synthetic_data):
            result = run(config)

        assert result is None

