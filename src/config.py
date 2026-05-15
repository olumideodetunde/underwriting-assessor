"""
Pydantic schema for config/*.yaml files.

Usage:
    from src.config import load_config
    config = load_config("config/renewal.yaml")   # returns a plain dict
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class FrequencyConfig(BaseModel):
    algorithm: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("algorithm")
    @classmethod
    def algorithm_must_be_known(cls, v: str) -> str:
        allowed = {"xgboost", "poisson_regressor", "gamma_regressor", "gradient_boosting"}
        if v not in allowed:
            raise ValueError(f"Unknown algorithm '{v}'. Choose from {sorted(allowed)}")
        return v


class TrackingConfig(BaseModel):
    uri: str = "http://127.0.0.1:5000"
    experiment_name: str
    run_name: str = ""
    run_description: str = ""
    artifact_model_name: str


class RunConfig(BaseModel):
    insurance_csv: str
    features: list[str] = Field(min_length=1)
    target: str
    frequency: FrequencyConfig
    tracking: TrackingConfig

    @field_validator("insurance_csv")
    @classmethod
    def csv_path_must_exist(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"Data file not found: {v}")
        return v


def load_config(yaml_path: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(yaml_path).read_text())
    validated = RunConfig(**raw)
    return validated.model_dump()