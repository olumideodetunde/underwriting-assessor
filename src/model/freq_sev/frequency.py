import mlflow
from xgboost import XGBRegressor
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

    # Handle NaN values - drop rows with NaN in training variables or target
    train_clean = train_with_eng_feature.dropna(subset=training_variables + [target])
    test_clean = test_with_eng_feature.dropna(subset=training_variables + [target])

    print(f"Training set: {len(train_with_eng_feature)} → {len(train_clean)} after dropping NaN")
    print(f"Test set: {len(test_with_eng_feature)} → {len(test_clean)} after dropping NaN")

    fig1 = plot_claims_distribution(train_clean)
    mlflow.set_tracking_uri(config["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["EXPERIMENT_NAME"])
    run_name = config.get("RUN_NAME", "")
    run_description = config.get("RUN_DESCRIPTION", "Poisson regression for insurance claims frequency prediction.")

    with mlflow.start_run(run_name=run_name, description=run_description) as run:
        run_id = run.info.run_id
        print(f"Started MLflow run: {run_id}")

        #train_dataset = mlflow.data.from_pandas(train_clean[training_variables], targets=target)
        #test_dataset = mlflow.data.from_pandas(test_clean[training_variables], targets=target)
        #mlflow.log_input(train_dataset, context="training")
        #mlflow.log_input(test_dataset, context="testing")

        xgb_regressor = XGBRegressor(
            objective='count:poisson',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model = xgb_regressor.fit(
            train_clean[training_variables],
            train_clean[target].values.ravel()
        )

        y_pred_train = xgb_model.predict(train_clean[training_variables])
        y_pred_test = xgb_model.predict(test_clean[training_variables])
        train_metrics = calculate_metrics(
            train_clean[target].values.ravel(),
            y_pred_train
        )
        test_metrics = calculate_metrics(
            test_clean[target].values.ravel(),
            y_pred_test
        )
        fig2 = plot_feature_importance(xgb_model, training_variables)
        fig3 = plot_actual_vs_predicted(
            test_clean[target].values.ravel(),
            y_pred_test
        )
        fig4 = plot_residuals(
            test_clean[target].values.ravel(),
            y_pred_test
        )
        mlflow.log_params(
            {
                'objective': xgb_model.objective,
                'n_estimators': xgb_model.n_estimators,
                'max_depth': xgb_model.max_depth,
                'learning_rate': xgb_model.learning_rate,
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
        mlflow.xgboost.log_model(
            xgb_model=xgb_model,
            artifact_path=config["ARTIFACT_MODEL_NAME"],
            input_example=test_clean[training_variables].head(5)
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

