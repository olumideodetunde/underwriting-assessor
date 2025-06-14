import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import psutil
from langchain.callbacks import FileCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentCallbackHandler(BaseCallbackHandler):
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics = {}
        self.start_time = None
        self.setup_directories()
        
    def setup_directories(self):
        os.makedirs("experiments", exist_ok=True)
        os.makedirs("experiments/prompts", exist_ok=True)
        os.makedirs("experiments/outputs", exist_ok=True)
        os.makedirs("experiments/metrics", exist_ok=True)
        
    def on_llm_start(self, *args, **kwargs):
        self.start_time = time.time()
        self.metrics["system_metrics_start"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
    def on_llm_end(self, *args, **kwargs):
        end_time = time.time()
        self.metrics["processing_time"] = end_time - self.start_time
        self.metrics["system_metrics_end"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        self.save_metrics()
        
    def save_metrics(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"experiments/metrics/{self.experiment_name}_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

def setup_langchain(
    model_name: str,
    temperature: float,
    max_tokens: int,
    experiment_name: str
) -> tuple[ChatOpenAI, ExperimentCallbackHandler]:
    callback_handler = ExperimentCallbackHandler(experiment_name)
    callback_manager = CallbackManager([callback_handler])
    
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        callback_manager=callback_manager
    )
    
    return llm, callback_handler

def save_prompt(
    prompt: str,
    prompt_name: str,
    experiment_name: str
) -> None:
    prompt_data = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment_name
    }
    
    prompt_file = f"experiments/prompts/{prompt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(prompt_file, "w") as f:
        json.dump(prompt_data, f, indent=2)
    logger.info(f"Prompt saved to {prompt_file}")

def save_output(
    output: str,
    output_name: str,
    experiment_name: str
) -> None:
    output_data = {
        "output": output,
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment_name
    }
    
    output_file = f"experiments/outputs/{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Output saved to {output_file}")

def run_experiment(
    experiment_name: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> None:
    try:
        llm, callback_handler = setup_langchain(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            experiment_name=experiment_name
        )
        
        save_prompt(user_prompt, "user_prompt", experiment_name)
        save_prompt(system_prompt, "system_prompt", experiment_name)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm(messages)
        output = response.content
        
        save_output(output, "llm_output", experiment_name)
        
        logger.info(f"Experiment '{experiment_name}' completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    experiment_name = "langchain_experiment_1"
    model_name = "gpt-3.5-turbo"
    system_prompt = "You are a helpful AI assistant."
    user_prompt = "Explain the concept of machine learning in simple terms."
    
    run_experiment(
        experiment_name=experiment_name,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=500
    )
