"""
Thin MLflow wrapper.
Every MLflow call in the project goes through this file.
To switch to W&B or another tracker, change only this file.
"""

import mlflow
from contextlib import contextmanager
from matplotlib.figure import Figure


def init(tracking_uri: str, experiment_name: str):
    """Call once at the start of a training script."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(run_name: str = "", description: str = ""):
    """Context manager that yields the run_id."""
    with mlflow.start_run(run_name=run_name, description=description) as run:
        yield run.info.run_id


def log_parameters(params: dict):
    """Log a flat dict of parameters."""
    mlflow.log_params(params)


def log_metrics_nested(metrics: dict, prefix: str = ""):
    """
    Log a dict of metric_name: value.
    prefix groups them, e.g. prefix="train" -> "train/mse", "train/mae".
    """
    for name, value in metrics.items():
        key = f"{prefix}/{name}" if prefix else name
        mlflow.log_metric(key, value)


def log_model(model, name: str, input_example=None):
    """
    Log any model. Detects the flavour automatically.
    - XGBoost models  -> mlflow.xgboost.log_model
    - Everything else -> mlflow.sklearn.log_model
    """
    from xgboost import XGBModel
    if isinstance(model, XGBModel):
        mlflow.xgboost.log_model(
            xgb_model=model,
            name=name,
            input_example=input_example,
        )
        return

    mlflow.sklearn.log_model(
        sk_model=model,
        name=name,
        input_example=input_example,
    )


def log_figures(figures: dict[str, Figure]):
    """
    Log a dict of filename: matplotlib figure.
    e.g. {"claims_distribution.png": fig1, "residuals.png": fig2}
    """
    for filename, fig in figures.items():
        mlflow.log_figure(fig, filename)

