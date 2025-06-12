import mlflow
import time
import psutil
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None
) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment '{experiment_name}' initialized")

def log_llm_parameters(
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    **kwargs
) -> None:
    params = {
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        **kwargs
    }
    mlflow.log_params(params)
    logger.info("LLM parameters logged successfully")

def log_system_metrics() -> Dict[str, float]:
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage('/').percent
    }
    mlflow.log_metrics(metrics)
    logger.info("System metrics logged successfully")
    return metrics

def log_prompt(
    prompt: str,
    prompt_name: str,
    prompt_version: str = "1.0"
) -> None:
    prompt_data = {
        "prompt": prompt,
        "version": prompt_version,
        "timestamp": datetime.now().isoformat()
    }
    
    os.makedirs("prompts", exist_ok=True)
    
    prompt_file = f"prompts/{prompt_name}_v{prompt_version}.json"
    with open(prompt_file, "w") as f:
        json.dump(prompt_data, f, indent=2)
    
    mlflow.log_artifact(prompt_file)
    logger.info(f"Prompt '{prompt_name}' logged successfully")

def log_dataset(
    input_data: List[Dict[str, Any]],
    dataset_name: str
) -> None:
    os.makedirs("datasets", exist_ok=True)
    
    dataset_file = f"datasets/{dataset_name}.json"
    with open(dataset_file, "w") as f:
        json.dump(input_data, f, indent=2)
    
    mlflow.log_artifact(dataset_file)
    logger.info(f"Dataset '{dataset_name}' logged successfully")

def log_model_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None
) -> None:
    mlflow.log_metrics(metrics, step=step)
    logger.info("Model metrics logged successfully")

def log_output_artifact(
    output_data: Any,
    artifact_name: str,
    artifact_type: str = "json"
) -> None:
    os.makedirs("artifacts", exist_ok=True)
    
    artifact_file = f"artifacts/{artifact_name}.{artifact_type}"
    
    if artifact_type == "json":
        with open(artifact_file, "w") as f:
            json.dump(output_data, f, indent=2)
    else:
        with open(artifact_file, "w") as f:
            f.write(str(output_data))
    
    mlflow.log_artifact(artifact_file)
    logger.info(f"Output artifact '{artifact_name}' logged successfully")

def run_experiment(
    experiment_name: str,
    model_name: str,
    prompt: str,
    input_data: List[Dict[str, Any]],
    **kwargs
) -> None:
    try:
        setup_mlflow(experiment_name)
        
        with mlflow.start_run():
            log_llm_parameters(
                model_name=model_name,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 100),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0)
            )
            
            initial_metrics = log_system_metrics()
            
            log_prompt(prompt, f"{model_name}_prompt")
            
            log_dataset(input_data, f"{model_name}_input")
            
            start_time = time.time()
            processing_time = time.time() - start_time
            
            log_model_metrics({
                "processing_time": processing_time,
                "input_size": len(input_data)
            })
            
            final_metrics = log_system_metrics()
            
            output_data = {
                "status": "success",
                "processing_time": processing_time,
                "system_metrics": {
                    "initial": initial_metrics,
                    "final": final_metrics
                }
            }
            log_output_artifact(output_data, f"{model_name}_output")
            
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    experiment_name = "llm_experiment_1"
    model_name = "gpt-3.5-turbo"
    prompt = "Analyze the following text: {text}"
    input_data = [
        {"text": "Sample text 1"},
        {"text": "Sample text 2"}
    ]
    
    run_experiment(
        experiment_name=experiment_name,
        model_name=model_name,
        prompt=prompt,
        input_data=input_data,
        temperature=0.8,
        max_tokens=150
    )
