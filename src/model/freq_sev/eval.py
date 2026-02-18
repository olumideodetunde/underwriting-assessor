import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
    median_absolute_error
)

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