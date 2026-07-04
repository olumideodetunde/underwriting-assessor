from src.data.loader import load_csv
from src.data.splitter import split_data
from src.feature import Driver, Vehicle
from src.model.factory import select_training_algorithm
from src.metrics import (plot_claims_distribution,
                         plot_feature_importance,
                         plot_residuals,
                         plot_actual_vs_predicted,
                         calculate_metrics)
from src import tracking


def train_and_evaluate(config) -> dict:
    """Train the frequency model and evaluate it, with no tracking side effects.

    Returns a dict with the fitted ``model``, ``train_metrics``/``test_metrics``,
    and the four diagnostic ``figures`` (keyed by their artifact filenames).
    Kept free of MLflow/S3 so it can be reused by the gate metrics smoke check
    (see ``scripts/gate_metrics_smoke.py``) without a tracking server or AWS creds.
    """
    insurance_dataset = load_csv(config['insurance_csv'])
    trainset, testset = split_data(insurance_dataset)

    driver = Driver()
    vehicle = Vehicle()

    trainset_feat = vehicle.fit(trainset).transform(trainset)
    trainset_feat = driver.transform(trainset_feat)

    testset_feat = vehicle.transform(testset)
    testset_feat = driver.transform(testset_feat)


    train_features = trainset_feat[config['features']]
    train_target = trainset_feat[config['target']]

    test_features = testset_feat[config['features']]
    test_target = testset_feat[config['target']]


    model = select_training_algorithm(
        config['frequency']['algorithm'],
        config['frequency']['parameters'],
    )
    model.fit(train_features, train_target.values.ravel())


    y_pred_train = model.predict(train_features)
    y_pred_test = model.predict(test_features)


    train_metrics = calculate_metrics(train_target, y_pred_train)
    test_metrics = calculate_metrics(test_target, y_pred_test)


    figures = {
        "claims_distribution.png": plot_claims_distribution(trainset_feat, target=config['target']),
        "feature_importance.png": plot_feature_importance(model, config['features']),
        "residuals.png": plot_residuals(test_target, y_pred_test),
        "actual_vs_predicted.png": plot_actual_vs_predicted(test_target, y_pred_test),
    }

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "figures": figures,
    }


def run(config):
    result = train_and_evaluate(config)

    tracking.init(config['tracking']['uri'], config['tracking']['experiment_name'])
    with tracking.start_run(
        run_name=config['tracking']['run_name'],
        description=config['tracking']['run_description'],
    ) as run_id:
        tracking.log_parameters(config['frequency']['parameters'])
        tracking.log_metrics_nested(result["train_metrics"], prefix="train")
        tracking.log_metrics_nested(result["test_metrics"], prefix="test")
        tracking.log_model(result["model"], name=config['tracking']['artifact_model_name'])
        tracking.log_figures(result["figures"])


if __name__ == '__main__':
    from src.config import load_config
    CONFIG = load_config(yaml_path='config/renewal.yaml')
    run(config=CONFIG)



