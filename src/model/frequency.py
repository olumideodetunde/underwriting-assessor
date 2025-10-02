import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model.dataset import main as dataset_prep_main
from model.feature import main as feature_eng_main
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
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
    "EXPERIMENT_NAME": "Insurance Claims Frequency Model-III",
    "MODEL_NAME": "poisson_model_v001",
    "ARTIFACT_MODEL_NAME": "poisson_model",
    "LOCAL_MODEL_PATH": "../../tmp",
    "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000",
    "INSURANCE_VARIABLES_PATH": "../../data/input/exp/Motor_vehicle_insurance_data.csv",
    "CLAIMS_VARIABLES_PATH": "../../data/input/exp/sample_type_claim.csv",
    "RUN_NAME": 'run_1_in_exp',
    "RUN_DESCRIPTION": "",

    "S3_BUCKET": os.getenv("MODEL_BUCKET_NAME"),
    "S3_BASE_PATH": os.getenv("MODEL_PATH", "dev"),
}

mlflow.set_tracking_uri(config["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(config["EXPERIMENT_NAME"])
insurance_variables_path = config["INSURANCE_VARIABLES_PATH"]
claims_variables_path = config["CLAIMS_VARIABLES_PATH"]


def plot_claims_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    df['claims_frequency'].hist(bins=30, ax=ax)
    ax.set_title('Claims Frequency Distribution')
    ax.set_xlabel('Claims Frequency')
    ax.set_ylabel('Count')
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
    ax.set_xlabel('Actual Claims Frequency')
    ax.set_ylabel('Predicted Claims Frequency')
    ax.set_title('Actual vs Predicted Claims Frequency')
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Claims Frequency')
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
    mask = y_pred > 0
    poisson_deviance = mean_poisson_deviance(y_true[mask], y_pred[mask])
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "medae": medae,
        "poisson_deviance": poisson_deviance
    }


def register_and_upload_model(config, run_id):
    """
    Register model and upload to S3

    Args:
        config: Configuration dictionary
        run_id: The MLflow run ID to register
    """
    client = MlflowClient()

    # Use the provided run_id instead of searching
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

    training_variables = config.get('TRAINING_VARIABLES',
                                    ['Car_age_years', 'Type_risk', 'Area', 'Value_vehicle', 'Distribution_channel'])
    target = config.get('TARGET', ['claims_frequency'])

    fig1 = plot_claims_distribution(train_with_eng_feature)

    mlflow.set_tracking_uri(config["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["EXPERIMENT_NAME"])

    run_name = config.get("RUN_NAME", "")
    run_description = config.get("RUN_DESCRIPTION", "Poisson regression for insurance claims frequency prediction.")

    with mlflow.start_run(run_name=run_name, description=run_description) as run:

        run_id = run.info.run_id
        print(f"Started MLflow run: {run_id}")

        poisson_regressor = PoissonRegressor(alpha=1e-12, solver='newton-cholesky', max_iter=300)
        poisson_model = poisson_regressor.fit(
            train_with_eng_feature[training_variables],
            train_with_eng_feature[target].values.ravel()
        )

        y_pred_train = poisson_model.predict(train_with_eng_feature[training_variables])
        y_pred_test = poisson_model.predict(test_with_eng_feature[training_variables])

        train_metrics = calculate_metrics(
            train_with_eng_feature[target].values.ravel(),
            y_pred_train
        )
        test_metrics = calculate_metrics(
            test_with_eng_feature[target].values.ravel(),
            y_pred_test
        )

        fig2 = plot_feature_importance(poisson_model, training_variables)
        fig3 = plot_actual_vs_predicted(
            test_with_eng_feature[target].values.ravel(),
            y_pred_test
        )
        fig4 = plot_residuals(
            test_with_eng_feature[target].values.ravel(),
            y_pred_test
        )

        mlflow.log_param("alpha", poisson_regressor.alpha)
        mlflow.log_param("solver", poisson_regressor.solver)
        mlflow.log_param("max_iter", poisson_regressor.max_iter)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("train_variables", training_variables)
        mlflow.log_param("model_name", config["MODEL_NAME"])

        for name, value in train_metrics.items():
            mlflow.log_metric(f"train/{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test/{name}", value)


        mlflow.sklearn.log_model(
            sk_model=poisson_model,
            artifact_path=config["ARTIFACT_MODEL_NAME"],  # Changed from 'name' to 'artifact_path'
            input_example=test_with_eng_feature[training_variables].head(5)
        )

        mlflow.log_figure(fig1, "claims_distribution.png")
        mlflow.log_figure(fig2, "feature_importance.png")
        mlflow.log_figure(fig3, "actual_vs_predicted.png")
        mlflow.log_figure(fig4, "residuals.png")

        print(f"✓ Model training completed")

    # After the run completes, register and upload the model
    print("\n" + "=" * 50)
    print("Registering and uploading model...")
    print("=" * 50)
    version = register_and_upload_model(config, run_id)

    print("\n" + "=" * 50)
    print(f"✓ Process completed successfully!")
    print(f"Model: {config['MODEL_NAME']} v{version}")
    print(f"Run ID: {run_id}")
    print("=" * 50)

    return run_id, version


if __name__ == "__main__":
    run_id, version = main(config)
