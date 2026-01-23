import joblib
from preprocessing import test_x
import numpy as np

# Load the trained pipeline
models = {"xgboost": "./xgb_pipeline_2026-01-23-15-50-25.joblib"}

  
# Make inference function
def make_inference(model_name, input_data):
    """
    Make inference using the trained model pipeline.

    Parameters:
    input_data (pd.DataFrame): Input data for prediction.

    Returns:
    np.ndarray: Predicted values.
    """
    model_path = models.get(model_name)
    if model_path is None:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
    model_pipeline = joblib.load(model_path) 
     
    predictions = model_pipeline.predict(input_data)
    predictions = np.expm1(predictions)  # Reverse log1p transformation if applied during training
    return print(predictions.round(2))  

if __name__ == "__main__":
    # Example usage
    make_inference("xgboost", test_x.iloc[:1]) # Predict on first 1 sample
    print("Inference module loaded successfully.")  
    
    