#####This script will be used for training and tracking the frequency model######

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

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Insurance Claims Frequency Model")

#####Define the input data#####
insurance_variables_path = "../../data/input/exp/Motor_vehicle_insurance_data.csv"
claims_variables_path = "../../data/input/exp/sample_type_claim.csv"

def plot_claims_distribution(df):
    """Plot claims frequency distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df['claims_frequency'].hist(bins=30, ax=ax)
    ax.set_title('Claims Frequency Distribution')
    ax.set_xlabel('Claims Frequency')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance/coefficients."""
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
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Claims Frequency')
    ax.set_ylabel('Predicted Claims Frequency')
    ax.set_title('Actual vs Predicted Claims Frequency')
    plt.tight_layout()
    return fig

def plot_residuals(y_true, y_pred):
    """Plot residuals."""
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
    """Calculate model performance metrics."""
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

def main():
    # Prepare dataset and engineer features
    train, test = dataset_prep_main(insurance_variables_path, claims_variables_path)
    train_with_eng_feature = feature_eng_main(train)
    test_with_eng_feature = feature_eng_main(test)

    # Define features and target
    training_variables = ['Car_age_years', 'Type_risk', 'Area', 'Value_vehicle',
                         'Distribution_channel']
    target = ['claims_frequency']

    # Generate pre-training visualizations
    fig1 = plot_claims_distribution(train_with_eng_feature)

    # Start MLflow run
    experiment_name = "Insurance Claims Frequency Model-I"
    run_name = "poisson_model_v1"
    run_description = "Poisson regression for insurance claims frequency prediction."

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, description=run_description):
        # Train model
        poisson_regressor = PoissonRegressor(alpha=1e-12, solver='newton-cholesky', max_iter=300)
        poisson_model = poisson_regressor.fit(
            train_with_eng_feature[training_variables],
            train_with_eng_feature[target].values.ravel()
        )

        # Make predictions
        y_pred_train = poisson_model.predict(train_with_eng_feature[training_variables])
        y_pred_test = poisson_model.predict(test_with_eng_feature[training_variables])

        # Calculate metrics
        train_metrics = calculate_metrics(
            train_with_eng_feature[target].values.ravel(),
            y_pred_train
        )
        test_metrics = calculate_metrics(
            test_with_eng_feature[target].values.ravel(),
            y_pred_test
        )

        # Generate post-training visualizations
        fig2 = plot_feature_importance(poisson_model, training_variables)
        fig3 = plot_actual_vs_predicted(
            test_with_eng_feature[target].values.ravel(),
            y_pred_test
        )
        fig4 = plot_residuals(
            test_with_eng_feature[target].values.ravel(),
            y_pred_test
        )

        # Log parameters
        mlflow.log_param("alpha", poisson_regressor.alpha)
        mlflow.log_param("solver", poisson_regressor.solver)
        mlflow.log_param("max_iter", poisson_regressor.max_iter)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("train_variables", training_variables)


        # Log metrics with train/test prefixes
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train/{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test/{name}", value)

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=poisson_model,
            artifact_path="poisson_model",
            input_example=test_with_eng_feature[training_variables]
        )
        mlflow.log_figure(fig1, "claims_distribution.png")
        mlflow.log_figure(fig2, "feature_importance.png")
        mlflow.log_figure(fig3, "actual_vs_predicted.png")
        mlflow.log_figure(fig4, "residuals.png")



if __name__ == "__main__":
    main()
    EXPERIMENT_NAME = "Insurance Claims Frequency Model-I"
    MODEL_NAME = "poisson_model_v1"
    from mlflow.tracking.client import MlflowClient
    # Get the latest run ID from the experiment
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1)
    run_id = runs[0].info.run_id
    print("Latest Run ID:", run_id)

    model_uri = f"runs:/{run_id}/model"
    # Register model
    registered_model = mlflow.register_model(model_uri, MODEL_NAME)

    import os
    version = registered_model.version
    LOCAL_MODEL_PATH = "."

    model_version_info = client.get_model_version(MODEL_NAME, version)

    # This downloads all model artifacts under the versioned model to the local path
    mlflow.artifacts.download_artifacts(
        artifact_uri=model_version_info.source,
        dst_path=LOCAL_MODEL_PATH
    )

    print(f"Model saved to: {os.path.abspath(LOCAL_MODEL_PATH)}")

    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Access the variables
    bucket_name = os.getenv("MODEL_BUCKET_NAME")
    model_path = os.getenv("MODEL_PATH")

    s3_bucket = f"s3://{bucket_name}/{model_path}"
    print(f"Uploading model to S3 bucket: {s3_bucket}")
    # Upload the model and its associated files to S3
    os.system(f"aws s3 cp \"{os.path.abspath(LOCAL_MODEL_PATH)}\model\" {s3_bucket}/{version} --recursive")
    # Also upload to the `latest`` folder in the bucket
    os.system(f"aws s3 cp \"{os.path.abspath(LOCAL_MODEL_PATH)}\model\" {s3_bucket}/latest --recursive")
    print(f"Model uploaded to S3: {s3_bucket}/{version}")

