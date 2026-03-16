import pytest
from unittest.mock import patch, MagicMock
from matplotlib.figure import Figure

from src import tracking


# --- Fixtures ---

@pytest.fixture
def mock_mlflow():
    with patch("src.tracking.mlflow") as mocked:
        mock_run = MagicMock()
        mock_run.info.run_id = "fake-run-id-123"
        mocked.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mocked.start_run.return_value.__exit__ = MagicMock(return_value=False)
        yield mocked


@pytest.fixture
def sample_figures():
    fig1 = Figure()
    fig2 = Figure()
    return {
        "distribution.png": fig1,
        "residuals.png": fig2,
    }


# =============================================================
# 1. init
# =============================================================

class TestInit:

    def test_sets_tracking_uri(self, mock_mlflow):
        tracking.init("http://localhost:5000", "test-experiment")
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_sets_experiment_name(self, mock_mlflow):
        tracking.init("http://localhost:5000", "test-experiment")
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")


# =============================================================
# 2. start_run
# =============================================================

class TestStartRun:

    def test_yields_run_id(self, mock_mlflow):
        with tracking.start_run(run_name="test-run") as run_id:
            assert run_id == "fake-run-id-123"

    def test_passes_run_name_and_description(self, mock_mlflow):
        with tracking.start_run(run_name="my-run", description="a description"):
            pass
        mock_mlflow.start_run.assert_called_once_with(
            run_name="my-run", description="a description"
        )

    def test_defaults_to_empty_strings(self, mock_mlflow):
        with tracking.start_run():
            pass
        mock_mlflow.start_run.assert_called_once_with(
            run_name="", description=""
        )


# =============================================================
# 3. log_params
# =============================================================

class TestLogParams:

    def test_logs_params_dict(self, mock_mlflow):
        params = {"n_estimators": 100, "max_depth": 6}
        tracking.log_parameters(params)
        mock_mlflow.log_params.assert_called_once_with(params)

    def test_logs_empty_dict(self, mock_mlflow):
        tracking.log_parameters({})
        mock_mlflow.log_params.assert_called_once_with({})


# =============================================================
# 4. log_metrics
# =============================================================

class TestLogMetrics:

    def test_adds_prefix_to_metric_names(self, mock_mlflow):
        metrics = {"mse": 0.5, "mae": 0.3}
        tracking.log_metrics_nested(metrics, prefix="train")
        mock_mlflow.log_metric.assert_any_call("train/mse", 0.5)
        mock_mlflow.log_metric.assert_any_call("train/mae", 0.3)

    def test_no_prefix_logs_bare_names(self, mock_mlflow):
        tracking.log_metrics_nested({"mse": 0.5})
        mock_mlflow.log_metric.assert_called_once_with("mse", 0.5)

    def test_empty_string_prefix_logs_bare_names(self, mock_mlflow):
        tracking.log_metrics_nested({"r2": 0.9}, prefix="")
        mock_mlflow.log_metric.assert_called_once_with("r2", 0.9)

    def test_logs_correct_number_of_metrics(self, mock_mlflow):
        metrics = {"mse": 0.5, "mae": 0.3, "r2": 0.9}
        tracking.log_metrics_nested(metrics, prefix="test")
        assert mock_mlflow.log_metric.call_count == 3


# =============================================================
# 5. log_model
# =============================================================

class TestLogModel:

    def test_xgboost_model_uses_xgboost_flavour(self, mock_mlflow):
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=10)
        tracking.log_model(model, name="model")
        mock_mlflow.xgboost.log_model.assert_called_once()
        mock_mlflow.sklearn.log_model.assert_not_called()

    def test_sklearn_model_uses_sklearn_flavour(self, mock_mlflow):
        from sklearn.linear_model import GammaRegressor
        model = GammaRegressor()
        tracking.log_model(model, name="model")
        mock_mlflow.sklearn.log_model.assert_called_once()
        mock_mlflow.xgboost.log_model.assert_not_called()

    def test_passes_artifact_path(self, mock_mlflow):
        from sklearn.linear_model import GammaRegressor
        model = GammaRegressor()
        tracking.log_model(model, name="gamma_model")
        mock_mlflow.sklearn.log_model.assert_called_once_with(
            sk_model=model,
            name="gamma_model",
            input_example=None,
        )

    def test_passes_input_example(self, mock_mlflow):
        from sklearn.linear_model import GammaRegressor
        model = GammaRegressor()
        example = [1, 2, 3]
        tracking.log_model(model, name="model", input_example=example)
        _, kwargs = mock_mlflow.sklearn.log_model.call_args
        assert kwargs["input_example"] == [1, 2, 3]


# =============================================================
# 6. log_figures
# =============================================================

class TestLogFigures:

    def test_logs_each_figure_with_filename(self, mock_mlflow, sample_figures):
        tracking.log_figures(sample_figures)
        assert mock_mlflow.log_figure.call_count == 2
        mock_mlflow.log_figure.assert_any_call(
            sample_figures["distribution.png"], "distribution.png"
        )
        mock_mlflow.log_figure.assert_any_call(
            sample_figures["residuals.png"], "residuals.png"
        )

    def test_empty_dict_logs_nothing(self, mock_mlflow):
        tracking.log_figures({})
        mock_mlflow.log_figure.assert_not_called()

