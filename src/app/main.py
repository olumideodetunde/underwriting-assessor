import logging
import os

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME")
MODEL_PATH = os.environ.get("LOAD_MODEL_PATH")

app = FastAPI(
    title = "Claims Frequency Inference",
    description="Predict claims frequency using Poisson Regressor trained on Spanish Insurance Motor dataset.",
    version="0.0.1",
)


model = None

class MotorInsuranceData(BaseModel):
    Car_age_years: int
    Type_risk: int
    Area: int
    Value_vehicle:float
    Distribution_channel:int

@app.on_event("startup")
def load_model():
    global model
    logger.info('Starting and loading MLFlow model...')
    try:
        model_uri = f"s3://{MODEL_BUCKET_NAME}/{MODEL_PATH}"
        logger.info(f"Loading MLFlow model from S3: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load MLFlow model: {e}")
        raise e

@app.post("/predict")
def predict(data: MotorInsuranceData):

    try:
        print(f"Received data: {data}")
        if model is None:
            raise HTTPException(status_code=500, detail="Model Not Loaded")
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ready")
def readiness_checkpoint():
    logging.info("Readiness Checkpoint Ready")
    if model is None:
        logger.error("Model Not Loaded")
        raise HTTPException(status_code=500, detail="Model Not Loaded")
    return {"ready": True}
