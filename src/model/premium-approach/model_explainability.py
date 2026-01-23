# Model Explainer Setup Script
## Install required packages
# !pip install shap

# Load pipelines and explainer libraries
import shap
import joblib
import pandas as pd
from preprocessing import test_x
from sklearn.pipeline import Pipeline
import numpy as np

#load the trained model pipeline
models = {"xgboost": "./xgb_pipeline_2026-01-23-15-50-25.joblib"}

def model_explainer(model_name, test_x):
    ############### SHAP Explainer #################
    
    model_pipeline = joblib.load(models[model_name])

    # Extract preprocessing steps from the pipeline
    preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])

    # Transform test data through Preprocessor
    X_transformed = preprocessor_pipeline.transform(test_x)  # shape: (n_samples, n_features_after_encoding)

    # Get feature names after feature selection
    preprocessor = model_pipeline.named_steps['preprocessor']
    all_features = preprocessor.get_feature_names_out()

    # Create DataFrame for SHAP
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_features)

    # # # SHAP
    model_name = list(model_pipeline.named_steps.keys())[-1]
    model = model_pipeline.named_steps[model_name]  # Get the final model step
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed_df)

    # Summary plot
    shap.summary_plot(shap_values, X_transformed_df)

    # Waterfall plot for a specific instance (e.g., index 1)
    shap.plots.waterfall(shap_values[0])
    
    shap.plots.bar(shap_values)
    

# Run the explainer
if __name__ == "__main__":  
    model_explainer("xgboost", test_x)