import os
import mlflow
from mlflow.tracking.client import MlflowClient
from dotenv import load_dotenv

load_dotenv()


def get_config():
    return {
        "EXPERIMENT_NAME": os.getenv("EXPERIMENT_NAME", "Insurance Claims Frequency Model-III"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "poisson_model_v001"),
        "ARTIFACT_MODEL_NAME": os.getenv("ARTIFACT_MODEL_NAME", "poisson_model"),
        "LOCAL_MODEL_PATH": os.getenv("LOCAL_MODEL_PATH", "../../tmp"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        "INSURANCE_VARIABLES_PATH": os.getenv("INSURANCE_VARIABLES_PATH", "../../data/input/exp/Motor_vehicle_insurance_data.csv"),
        "CLAIMS_VARIABLES_PATH": os.getenv("CLAIMS_VARIABLES_PATH", "../../data/input/exp/sample_type_claim.csv"),
        "RUN_NAME": os.getenv("RUN_NAME", "run_1_in_exp"),
        "RUN_DESCRIPTION": os.getenv("RUN_DESCRIPTION", "Poisson regression for insurance claims frequency prediction"),
        "S3_BUCKET": os.getenv("MODEL_BUCKET_NAME"),
        "S3_BASE_PATH": os.getenv("MODEL_PATH", "dev")
    }


def register_and_upload_model(config, run_id):

    client = MlflowClient()
    print(f"Registering model from run: {run_id}")
    model_uri = f"runs:/{run_id}/{config['ARTIFACT_MODEL_NAME']}"
    registered_model = mlflow.register_model(model_uri, config["MODEL_NAME"])
    version = registered_model.version
    print(f"Model registered as {config['MODEL_NAME']} version {version}")
    model_version_info = client.get_model_version(config["MODEL_NAME"], version)
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_version_info.source,
        dst_path=config["LOCAL_MODEL_PATH"]
    )
    print(f"Model downloaded to: {os.path.abspath(local_path)}")

    # Upload to S3 if bucket is configured
    if config["S3_BUCKET"]:
        # Construct S3 path - using version number for organization
        s3_path = f"{config['S3_BASE_PATH']}/{version}"
        s3_full_path = f"s3://{config['S3_BUCKET']}/{s3_path}"

        print(f"Uploading model to S3: {s3_full_path}")
        print(f"Local model path: {local_path}")

        # Upload to S3
        result = os.system(f"aws s3 cp '{local_path}' {s3_full_path} --recursive")
        if result == 0:
            print(f"✓ Model successfully uploaded to: {s3_full_path}")
            print(f"To use this model, set MODEL_PATH={s3_path}")
        else:
            print(f"✗ Failed to upload model to S3")
    else:
        print("S3_BUCKET not configured, skipping S3 upload")
    return version
