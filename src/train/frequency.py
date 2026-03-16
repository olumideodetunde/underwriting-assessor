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


def run(config):
    insurance_dataset = load_csv(config['insurance_csv'])
    trainset, testset = split_data(insurance_dataset)

    driver = Driver()
    vehicle = Vehicle()

    trainset_feat = vehicle.transform(trainset)
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


    fig_dist = plot_claims_distribution(trainset_feat, target=config['target'])
    fig_importance = plot_feature_importance(model, config['features'])
    fig_residuals = plot_residuals(test_target, y_pred_test)
    fig_actual_pred = plot_actual_vs_predicted(test_target, y_pred_test)


    tracking.init(config['tracking']['uri'], config['tracking']['experiment_name'])

    with tracking.start_run(
        run_name=config['tracking']['run_name'],
        description=config['tracking']['run_description'],
    ) as run_id:
        tracking.log_params(config['frequency']['parameters'])
        tracking.log_metrics(train_metrics, prefix="train")
        tracking.log_metrics(test_metrics, prefix="test")
        tracking.log_model(model, artifact_path=config['tracking']['artifact_model_name'])
        tracking.log_figures({
            "claims_distribution.png": fig_dist,
            "feature_importance.png": fig_importance,
            "residuals.png": fig_residuals,
            "actual_vs_predicted.png": fig_actual_pred,
        })


if __name__ == '__main__':
    from src.config import load_config
    CONFIG = load_config(yaml_path='config/renewal.yaml')
    run(config=CONFIG)



