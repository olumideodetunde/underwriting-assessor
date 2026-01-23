## install required packages
# !pip install category_encoders --quiet
# !pip install xgboost --quiet
# !pip install optuna --quiet

## import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from feature_eng import feature_engineering
from sklearn.metrics import  root_mean_squared_error
from sklearn.pipeline import make_pipeline
import optuna
from xgboost import XGBRegressor
import joblib
from feature_eng import feature_engineering
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessor,train_x, test_x, train_y, test_y


## Use Hyperparameter optimization with Optuna and 10-fold cross-validation - XGBRegressor

# -------------------------
# Optuna objective
# -------------------------
def objective(trial):

    preprocessor = Preprocessor(
        target_smoothing=trial.suggest_float(
            "target_smoothing", 0.3, 5.0, log=True
        ),
        min_samples_leaf=trial.suggest_int(
            "te_min_samples_leaf", 20, 80
        ),
    )

    # Define XGBRegressor hyperparameters to tune
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist"
    }

    # Create the model
    model = XGBRegressor(**params)

    pipeline = make_pipeline(preprocessor, model)

    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        train_x,
        train_y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1  # parallelize only at CV level
    )

    return -scores.mean()

# -------------------------
# Run Optuna study
# -------------------------
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

# -------------------------
# Print best hyperparameters
# -------------------------
print("Best CV RMSE:", study.best_value)
print("Best parameters:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")

best = study.best_params

# -------------------------
# Build final leak-proof pipeline using **params
# -------------------------
final_preprocessor = Preprocessor(
    target_smoothing=best["target_smoothing"],
    min_samples_leaf=best["te_min_samples_leaf"],
)

final_params = {
    "n_estimators": best["n_estimators"],
    "max_depth": best["max_depth"],
    "learning_rate": best["learning_rate"],
    "subsample": best["subsample"],
    "colsample_bytree": best["colsample_bytree"],
    "reg_alpha": best["reg_alpha"],
    "reg_lambda": best["reg_lambda"],
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist"
}



# Final model
final_model = XGBRegressor(**final_params)

final_pipeline = make_pipeline(
    final_preprocessor,
    final_model
)

# -------------------------
# Train final model
# -------------------------
final_pipeline.fit(train_x, train_y)

# Evaluate on test set
test_preds = final_pipeline.predict(test_x)
test_rmse = root_mean_squared_error(test_y, test_preds)
print("Final test RMSE:", test_rmse)

# Save final model
joblib.dump(
    final_pipeline,
    f"xgb_pipeline_{pd.Timestamp.today().strftime('%Y-%m-%d-%H-%M-%S')}.joblib"
)

if __name__ == "__main__":
    print("Final XGB model saved successfully.")