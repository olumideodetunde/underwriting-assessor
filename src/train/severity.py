import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.model.dataset import main as dataset_prep_main
from src.model.feature import main as feature_eng_main
from sklearn.linear_model import GammaRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)
from mlflow.tracking.client import MlflowClient
import os
from dotenv import load_dotenv

# Load environment variables at the top
load_dotenv()

config = {
    "EXPERIMENT_NAME": "Insurance Claims Severity Model-II",
    "MODEL_NAME": "gamma_model_v001",
    "ARTIFACT_MODEL_NAME": "gamma_model",
    "LOCAL_MODEL_PATH": os.path.abspath("tmp"),
    "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000",
    "INSURANCE_VARIABLES_PATH": os.path.abspath("data/input/exp/Insurance_Initiation_Variables.csv"),
    "CLAIMS_VARIABLES_PATH": os.path.abspath("data/input/exp/sample_type_claim.csv"),
    "RUN_NAME": 'run_1_in_exp',
    "RUN_DESCRIPTION": "",
    "TRAINING_VARIABLES": ['Car_age_years', 'Type_risk', 'Area', 'Value_vehicle', 'Distribution_channel', 'Cylinder_capacity'],
    "TARGET": "Cost_claims_year",
    "S3_BUCKET": os.getenv("MODEL_BUCKET_NAME"),
    "S3_BASE_PATH": os.getenv("MODEL_PATH", "dev"),
}

mlflow.set_tracking_uri(config["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(config["EXPERIMENT_NAME"])

def plot_loss_distribution(df, target):
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 5))
    ax0.set_title('Loss Distribution')
    df[target].hist(bins=40, log=True, ax=ax0)

    p2_5, p97_5 = np.percentile(df[target], [2.5, 97.5])
    middle_95 = df[target][(df[target] >= p2_5) & (df[target] <= p97_5)]
    ax1.set_title('Middle-95% Loss Distribution (2.5%-97.5%)')
    middle_95.hist(bins=40, log=False, ax=ax1)
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    coefficients = model.coef_ if hasattr(model, 'coef_') else None
    if coefficients is not None:
        coef_df = pd.DataFrame({'feature': feature_names, 'importance': coefficients})
        coef_df = coef_df.sort_values('importance', ascending=True)
        sns.barplot(data=coef_df, x='importance', y='feature', ax=ax)
        ax.set_title('Feature Importance')
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Claims Severity')
    ax.set_ylabel('Predicted Claims Severity')
    ax.set_title('Actual vs Predicted Claims Severity')
    plt.tight_layout()
    return fig

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Claims Severity')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    plt.tight_layout()
    return fig

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "medae": medae,
    }

def register_and_upload_model(config, run_id):
    """
    Register model and upload to S3

    Args:
        config: Configuration dictionary
        run_id: The MLflow run ID to register
    """
    client = MlflowClient()
    print(f"Registering model from run: {run_id}")

    # Register the model
    model_uri = f"runs:/{run_id}/{config['ARTIFACT_MODEL_NAME']}"
    registered_model = mlflow.register_model(model_uri, config["MODEL_NAME"])
    version = registered_model.version

    print(f"Model registered as {config['MODEL_NAME']} version {version}")

    # Download the model locally
    model_version_info = client.get_model_version(config["MODEL_NAME"], version)
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_version_info.source,
        dst_path=config["LOCAL_MODEL_PATH"]
    )
    print(f"Model downloaded to: {os.path.abspath(local_path)}")

    # Upload to S3 if bucket is configured
    if config["S3_BUCKET"]:
        # Construct S3 path - using version number for organization
        s3_path = f"{config['S3_BASE_PATH']}/{version}"
        s3_full_path = f"s3://{config['S3_BUCKET']}/{s3_path}"

        print(f"Uploading model to S3: {s3_full_path}")
        print(f"Local model path: {local_path}")

        # Upload to S3
        result = os.system(f"aws s3 cp '{local_path}' {s3_full_path} --recursive")

        if result == 0:
            print(f"✓ Model successfully uploaded to: {s3_full_path}")
            print(f"To use this model, set MODEL_PATH={s3_path}")
        else:
            print(f"✗ Failed to upload model to S3")
    else:
        print("S3_BUCKET not configured, skipping S3 upload")

    return version

def main(config):
    train, test = dataset_prep_main(config["INSURANCE_VARIABLES_PATH"], config["CLAIMS_VARIABLES_PATH"])
    train_with_eng_feature = feature_eng_main(train)
    test_with_eng_feature = feature_eng_main(test)

    training_variables = config["TRAINING_VARIABLES"]
    target = config["TARGET"]

    # Filter for positive claims only (gamma distribution requirement)
    train_mask = train_with_eng_feature[target] > 0
    train_with_eng_feature = train_with_eng_feature[train_mask]
    test_mask = test_with_eng_feature[target] > 0
    test_with_eng_feature = test_with_eng_feature[test_mask]

    fig1 = plot_loss_distribution(train_with_eng_feature, target)

    mlflow.set_tracking_uri(config["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["EXPERIMENT_NAME"])

    run_name = config.get("RUN_NAME", "")
    run_description = config.get("RUN_DESCRIPTION", "Gamma regression for insurance claims severity prediction.")

    with mlflow.start_run(run_name=run_name, description=run_description) as run:
        run_id = run.info.run_id
        print(f"Started MLflow run: {run_id}")

        # Train baseline model
        dummy_regressor = DummyRegressor()
        dummy_model = dummy_regressor.fit(
            train_with_eng_feature[training_variables],
            train_with_eng_feature[target]
        )

        # Train Gamma model
        gamma_regressor = GammaRegressor(alpha=10, solver="newton-cholesky")
        gamma_model = gamma_regressor.fit(
            train_with_eng_feature[training_variables],
            train_with_eng_feature[target]
        )

        # Make predictions
        y_pred_train_dummy = dummy_model.predict(train_with_eng_feature[training_variables])
        y_pred_test_dummy = dummy_model.predict(test_with_eng_feature[training_variables])
        y_pred_train = gamma_model.predict(train_with_eng_feature[training_variables])
        y_pred_test = gamma_model.predict(test_with_eng_feature[training_variables])

        # Calculate metrics
        train_metrics_dummy = calculate_metrics(
            train_with_eng_feature[target],
            y_pred_train_dummy
        )
        test_metrics_dummy = calculate_metrics(
            test_with_eng_feature[target],
            y_pred_test_dummy
        )
        train_metrics = calculate_metrics(
            train_with_eng_feature[target],
            y_pred_train
        )
        test_metrics = calculate_metrics(
            test_with_eng_feature[target],
            y_pred_test
        )

        # Create plots
        fig2 = plot_feature_importance(gamma_model, training_variables)
        fig3 = plot_actual_vs_predicted(
            test_with_eng_feature[target],
            y_pred_test
        )
        fig4 = plot_residuals(
            test_with_eng_feature[target],
            y_pred_test
        )

        # Log parameters
        mlflow.log_param("alpha", gamma_regressor.alpha)
        mlflow.log_param("solver", gamma_regressor.solver)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("train_variables", training_variables)
        mlflow.log_param("model_name", config["MODEL_NAME"])

        # Log metrics
        for name, value in train_metrics_dummy.items():
            mlflow.log_metric(f"train_dummy/{name}", value)
        for name, value in test_metrics_dummy.items():
            mlflow.log_metric(f"test_dummy/{name}", value)
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train/{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test/{name}", value)

        # Log model
        mlflow.sklearn.log_model(
            gamma_model,
            config["ARTIFACT_MODEL_NAME"],
            registered_model_name=config["MODEL_NAME"]
        )

        # Log figures
        mlflow.log_figure(fig1, "loss_distribution.png")
        mlflow.log_figure(fig2, "feature_importance.png")
        mlflow.log_figure(fig3, "actual_vs_predicted.png")
        mlflow.log_figure(fig4, "residuals.png")

        version = register_and_upload_model(config, run_id)
        return version

if __name__ == "__main__":
    main(config)
