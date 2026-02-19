import os

import mlflow
from sklearn.linear_model import PoissonRegressor
from src.model.freq_sev.dataset import main as dataset_prep_main
from src.model.freq_sev.feature import main as feature_eng_main
from src.model.freq_sev.utils import register_and_upload_model, get_frequency_config
from src.model.freq_sev.eval import (plot_claims_distribution,
                                     plot_feature_importance,
                                     plot_residuals,
                                     plot_actual_vs_predicted,
                                     calculate_metrics)
CONFIG = get_frequency_config()
mlflow.set_tracking_uri(CONFIG["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(CONFIG["EXPERIMENT_NAME"])
insurance_variables_path = CONFIG["INSURANCE_VARIABLES_PATH"]
claims_variables_path = CONFIG["CLAIMS_VARIABLES_PATH"]

def main(config):
    train, test = dataset_prep_main(config["INSURANCE_VARIABLES_PATH"], config["CLAIMS_VARIABLES_PATH"])
    train_with_eng_feature = feature_eng_main(train)
    test_with_eng_feature = feature_eng_main(test)
    training_variables = config.get('FREQ_FEATURES').split(',')
    target = config.get('FREQ_TARGET')


    fig1 = plot_claims_distribution(train_with_eng_feature)
    mlflow.set_tracking_uri(config["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["EXPERIMENT_NAME"])
    run_name = config.get("RUN_NAME", "")
    run_description = config.get("RUN_DESCRIPTION", "Poisson regression for insurance claims frequency prediction.")

    with mlflow.start_run(run_name=run_name, description=run_description) as run:
        run_id = run.info.run_id
        print(f"Started MLflow run: {run_id}")

        train_dataset = mlflow.data.from_pandas(train_with_eng_feature, targets=target)
        test_dataset = mlflow.data.from_pandas(test_with_eng_feature, targets=target)
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(test_dataset, context="testing")


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
        mlflow.log_params(
            {
                'alpha': poisson_model.alpha,
                'solver': poisson_model.solver,
                'max_iter': poisson_model.max_iter,
                'run_name': run_name,
                'training_variables': training_variables,
                'target': target,
                'model_name': config["MODEL_NAME"],

            }
        )
        for name, value in train_metrics.items():
            mlflow.log_metric(f"train/{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test/{name}", value)
        mlflow.sklearn.log_model(
            sk_model=poisson_model,
            name=config["ARTIFACT_MODEL_NAME"],
            input_example=test_with_eng_feature[training_variables].head(5)
        )
        mlflow.log_figure(fig1, "claims_distribution.png")
        mlflow.log_figure(fig2, "feature_importance.png")
        mlflow.log_figure(fig3, "actual_vs_predicted.png")
        mlflow.log_figure(fig4, "residuals.png")
        print(f"✓ Model training completed")

    # After the run completes, register and upload the model
    # print("\n" + "=" * 50)
    # print("Registering and uploading model...")
    # print("=" * 50)
    # version = register_and_upload_model(config, run_id)
    # print("\n" + "=" * 50)
    # print(f"✓ Process completed successfully!")
    # print(f"Model: {config['MODEL_NAME']} v{version}")
    # print(f"Run ID: {run_id}")
    # print("=" * 50)
    return run_id  #, version
if __name__ == "__main__":
    run_id = main(CONFIG)

