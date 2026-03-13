# Underwriting Assessor

## The Business Problem

Insurance underwriting is the process of evaluating risk to determine whether to insure a client and at what price. This traditionally relies on manual assessment by underwriters using actuarial tables, experience, and judgment. It can be slow, inconsistent across underwriters, and hard to scale as application volumes grow.

This project builds a machine learning tool that predicts insurance premiums based on historical pricing data and risk factors. It sits alongside underwriter judgment, providing data-driven premium estimates to help with consistency and speed in the pricing workflow.

### Why This Is Hard: Two Distinct Pricing Problems

This project views pricing as 2 separate workflows:

| Scenario | What You Know | What Makes It Different |
|----------|---------------|------------------------|
| **New Client** | Demographics, vehicle info, coverage request | No claims history, no loyalty signal. You're pricing blind risk |
| **Renewal** | Everything above + claims history, tenure, prior premiums | You have behavioural data, but the client can leave if mispriced |

New client pricing is about risk estimation under uncertainty. Renewal pricing is about risk re-evaluation with evidence. The features, error tolerances, and business consequences differ enough that a single model struggles with both.

This project has separate model pipelines for each scenario. They share common infrastructure (data loading, feature engineering, storage, monitoring) but have independent training, evaluation, and serving paths. A YAML config file is what differs between the two.

### What This Project Delivers

```
config/renewal.yaml (or new_client.yaml)
        |
        v
scripts/train.py --config config/renewal.yaml
        |
        |-> src/config.py          (validate config)
        |-> src/data/loader.py     (load raw data)
        |-> src/data/splitter.py   (train/test split)
        |-> src/features/          (apply feature pipelines from config)
        |-> src/model/factory.py   (create model from config string)
        |-> src/training/trainer.py (train + cross-validate)
        |-> src/metrics.py         (evaluate model performance)
        |-> src/tracking.py        (log to MLflow)
        +-> src/storage.py         (save to S3)
                |
                v
src/serving/app.py
        |
        |-> POST /predict/new_client  -> new client model
        |-> POST /predict/renewal     -> renewal model
        +-> GET  /health              -> status check
```

**1. Dual Model Pipelines.** Separate training and inference paths for new client and renewal pricing, driven by YAML configuration files in `config/`.

**2. Feature Engineering.** Domain-driven feature transformations (driver age, driving experience, vehicle age, power-to-weight ratio) defined once and reused across training and serving to avoid training-serving skew.

**3. Model Training with Experiment Tracking.** MLflow integration logs parameters, metrics, and model artifacts for every training run so experiments are reproducible and comparable.

**4. Model Serving.** A FastAPI application with separate endpoints for new client and renewal predictions, input validation via Pydantic schemas, and health checks.

**5. Evaluation Metrics.** Classification, regression, and calibration metrics computed consistently across both model types.

**6. Monitoring.** Drift detection that compares live prediction distributions against training baselines to catch data quality issues and model degradation before they affect pricing.

**7. Testing.** Unit tests covering data loading, feature transformations, model training, and inference paths using pytest.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [UV](https://github.com/astral-sh/uv) package installer

### Setup

```bash
git clone https://github.com/olumideodetunde/underwriting-assessor.git
cd underwriting-assessor

pip install uv

uv venv
source .venv/bin/activate
uv pip install -e ".[test]"
```

### Train a Model

```bash
make train-new        # Train the new client model
make train-renewal    # Train the renewal model
make train-all        # Train both
```

Or directly:

```bash
python scripts/train.py --config config/new_client.yaml
python scripts/train.py --config config/renewal.yaml
```

### Serve Predictions

```bash
make serve
```

Or directly:

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Test a prediction:

```bash
curl -X POST http://localhost:8000/predict/new_client \
  -H "Content-Type: application/json" \
  -d '{"driver_age": 35, "vehicle_value": 25000, "area": 1, "type_risk": 3}'
```

### Run Tests

```bash
make test
```

Or directly:

```bash
pytest tests/ -v
pytest --cov=src
```

### Run with Docker

```bash
docker build -t underwriting-assessor .
docker run -p 8000:8000 underwriting-assessor
```

---

## Project Structure

```
underwriting-assessor/
|
|-- config/
|   |-- base.yaml                    # Shared defaults
|   |-- new_client.yaml              # Overrides for new clients
|   +-- renewal.yaml                 # Overrides for renewals
|
|-- data/
|   +-- input/                       # Raw data (gitignored)
|       +-- exp/
|
|-- logs/
|   +-- learning/                    # Step-by-step learning walkthroughs
|
|-- notebook/
|   |-- new_client/                  # EDA & feature exploration for new clients
|   +-- renewal/                     # EDA & feature exploration for renewals
|
|-- src/
|   |-- __init__.py
|   |-- config.py                    # Pydantic config schema
|   |-- tracking.py                  # MLflow experiment tracking wrapper
|   |-- storage.py                   # S3 model artifact storage
|   |-- monitoring.py                # Drift detection utilities
|   |-- metrics.py                   # Evaluation metric computation
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   |-- loader.py                # Load raw data, handle CSV quirks
|   |   +-- splitter.py              # Train/test split logic
|   |
|   |-- features/
|   |   |-- __init__.py              # Feature registry
|   |   |-- base.py                  # Abstract transformer contract
|   |   |-- driver.py                # Driver features (age, experience)
|   |   |-- payment.py               # Payment features
|   |   |-- claims.py                # Claims features
|   |   +-- vehicle.py               # Vehicle features
|   |
|   |-- model/
|   |   |-- __init__.py
|   |   +-- factory.py               # Model registry + creation from config
|   |
|   |-- training/
|   |   |-- __init__.py
|   |   +-- trainer.py               # Training orchestration
|   |
|   +-- serving/
|       |-- __init__.py
|       |-- app.py                   # FastAPI app + lifespan
|       |-- routes.py                # /predict/new_client, /predict/renewal, /health
|       |-- schemas.py               # Pydantic request/response models
|       +-- request_logger.py        # Middleware for latency/request counts
|
|-- scripts/
|   |-- train.py                     # CLI entry point for training
|   |-- deploy.py                    # Push model to S3 + update serving
|   +-- monitor.py                   # Run drift checks offline
|
|-- tests/
|   |-- test_features.py
|   |-- test_model.py
|   |-- test_training.py
|   +-- test_serving.py
|
|-- terraform/
|   |-- app/                         # AWS ECS, ALB, ECR resources
|   +-- setup/                       # VPC, IAM, S3 bootstrap
|
|-- Dockerfile
|-- docker-compose.yaml
|-- Makefile
|-- pyproject.toml
+-- uv.lock
```

---

## Learning Logs

The `logs/learning/` directory contains step-by-step walkthroughs of the key stages of this project, written for learning purposes. See `logs/README.md` for the full reading guide. The logs cover:

1. **Dataset Creation** -- merging policy data with claims history
2. **Exploration** -- EDA on claims frequency and severity distributions
3. **Feature Engineering** -- transforming raw data into model-ready features
4. **Frequency Modelling** -- predicting how often claims occur (Poisson regression)
5. **Severity Modelling** -- predicting how costly claims are (Gamma regression)

---

## Development

### Tools

| Tool | Purpose |
|------|---------|
| **UV** | Dependency management. Faster than pip/poetry, deterministic lock files |
| **pytest** | Testing framework with coverage reporting |
| **MLflow** | Experiment tracking. Logs params, metrics, and artifacts per run |
| **FastAPI** | Model serving with automatic OpenAPI docs |
| **Docker** | Consistent packaging across environments |
| **Pydantic** | Input validation for API requests and config files |
| **Terraform** | Infrastructure as code for AWS resources |

### Design Principles

- **Config drives behaviour.** The same training code handles both new client and renewal models. The YAML config determines which features, model type, and hyperparameters to use.

- **Scripts are thin entry points.** Every file in `scripts/` parses arguments and calls functions from `src/`. Logic lives in `src/`, invocation lives in `scripts/`.

- **Start flat, promote to directories when needed.** Files at the `src/` level (`config.py`, `monitoring.py`, `metrics.py`) are shared utilities used by training, serving, and scripts. They get promoted to directories when they outgrow a single file.

- **Notebooks explore, `src/` implements.** Exploratory analysis lives in `notebook/`. Production code lives in `src/` with abstractions, error handling, and tests.

- **Reproducibility over speed.** UV lock files, MLflow tracking, and versioned model artifacts mean training runs can be recreated.

---

## Planned Improvements

| Phase | What | Status |
|-------|------|--------|
| Core ML pipeline | Data loading, features, model training, evaluation | Done |
| Experiment tracking | MLflow integration | In progress |
| Model serving | FastAPI with dual endpoints | In progress |
| Containerization | Docker packaging | Planned |
| Cloud deployment | AWS ECS with Fargate | Planned |
| Infrastructure as code | Terraform for AWS resources | Planned |
| CI/CD | GitHub Actions for test + deploy | Planned |
| Monitoring | Drift detection in production | Planned |
