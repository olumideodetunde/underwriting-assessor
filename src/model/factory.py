from xgboost import XGBRegressor
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

def select_training_algorithm(algorithm_name:str, model_params:dict):
    models = {
        'xgboost': XGBRegressor,
        'poisson_regressor': PoissonRegressor,
        'gamma_regressor': GammaRegressor,
        'gradient_boosting': HistGradientBoostingRegressor,
    }
    return models[algorithm_name](**model_params)
