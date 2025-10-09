# Underwriting Assessor


In this project, I built a machine learning solution to predict insurance premiums and assist underwriters in risk assessment and pricing decisions. The goal is to leverage historical insurance data to build a regression model that estimates technical premiums based on risk factors like policy details, insured attributes, and coverage specifications. I use Python tooling and practices to create a testable and reproducible ML pipeline.

The insurance underwriting process relies on manual assessment and actuarial tables. This project demonstrates how machine learning can augment that process by learning patterns from historical pricing data to provide premium estimates. The model doesn't replace underwriter judgment but serves as a decision support tool that can improve consistency and speed in the pricing workflow.

This solution is useful for insurance teams looking to modernize their underwriting process without requiring infrastructure setup. The focus here is on building a foundation with data processing, feature engineering, and model training that can later be extended into a production deployment pipeline.

### High-Level Approach

In this project, I focus on building the core ML components with emphasis on code quality and reproducibility using the tools and practices below.

1. I use **UV** as the package installer instead of pip or poetry. UV is faster at resolving dependencies and creating virtual environments, which speeds up local development and CI/CD pipelines. It produces a lock file that ensures everyone on the team has identical dependencies, preventing the "works on my machine" problem.

2. The codebase follows a **modular architecture** with separation of concerns. Data processing logic lives in `dataset.py`, feature engineering in `feature.py`, model training in `model.py`, and inference logic in `inference.py`. This separation makes the code easier to test, maintain, and extend. Each module has a single responsibility.

3. I implement **unit tests** using pytest to validate each component independently. Testing is important in ML projects because bugs in data processing or feature engineering can degrade model performance. The test suite covers data loading, feature transformations, model training, and inference paths.

4. **Feature engineering** is treated as a component with dedicated implementation and testing. Insurance pricing depends on domain features, and getting this right is often more impactful than model selection. The feature pipeline transforms raw insurance data into predictive signals that capture risk patterns.

5. The project uses **development mode installation** which allows you to edit source code without reinstalling the package. This speeds up the development cycle as you can modify code, run tests, and iterate without package management overhead.

### What This Project Delivers

This repository contains an ML training pipeline for insurance premium prediction, from raw data to trained model artifacts. Here's what has been built:

**1. Data Processing Pipeline**

The data processing module handles loading and cleaning historical insurance datasets. It reads input data from the `data/input/` directory, performs validation to catch data quality issues early, and prepares the dataset for feature engineering. The pipeline handles data issues like missing values, outliers, and inconsistent formatting that appear in insurance data.

**2. Feature Engineering**

Feature engineering transforms raw insurance attributes into predictors. This involves creating derived features that capture risk signals, encoding categorical variables, and scaling numerical features. The feature engineering logic is centralized in a dedicated module, making it easy to add new features or modify existing ones without touching model code.

**3. Model Training**

The model training module implements the regression algorithm for premium prediction. It handles model selection, hyperparameter tuning, and evaluation metrics calculation. Trained models and their metadata are saved to the `data/output/` directory, making it easy to track different experiments and compare model performance over time.

**4. Inference Pipeline**

The inference module loads trained models and generates predictions for new insurance applications. It ensures that the same feature transformations used during training are applied at inference time, preventing training-serving skew. This component serves as the foundation for exposing the model through an API.

**5. Testing Infrastructure**

Unit tests validate each component's behavior. Tests cover edge cases, data validation, feature transformation correctness, and model inference paths. The test suite uses pytest with coverage reporting, making it easy to identify untested code paths and maintain code quality as the project evolves.

### Development Philosophy

This project prioritizes **reproducibility and maintainability** over quick prototyping. The structured approach here provides several advantages:

**Version Control**: All code is properly versioned, making it easy to track changes and collaborate with team members.

**Testability**: Modular code with unit tests catches bugs early and gives confidence when refactoring.

**Extensibility**: The architecture makes it straightforward to add new features, swap models, or integrate with production systems.

**Documentation**: The project structure and separation of concerns makes it easier for new team members to understand the codebase.

The notebook directory contains exploratory analysis and experimentation, but all production code lives in the `src/` directory with proper abstractions and error handling. This separation keeps exploration flexible while maintaining code quality in the core implementation.

### Current State and Future Direction

Right now, this project focuses on the training phase of the ML lifecycle. The model trains on historical data, evaluates performance, and saves artifacts for later use. This is a foundation, but it's just the first step in an MLOps pipeline.

The next steps would be:

**Experiment Tracking**: Integrate MLflow to log training runs, parameters, metrics, and model versions systematically.

**Model Serving**: Wrap the inference logic in a FastAPI service with health checks and input validation.

**Containerization**: Package the application in Docker for consistent deployment across environments.

**Cloud Deployment**: Deploy to AWS using ECS with Fargate for serverless, scalable inference.

**Infrastructure as Code**: Use Terraform to define and manage cloud resources reproducibly.

These extensions would transform the project from a local training pipeline into a production ML service. However, getting the foundational pieces right is the first step, and that's what this repository delivers.


---------------------
## Project Overview

This project implements a data-driven approach to insurance pricing prediction using:
- Feature engineering from historical insurance data
- Machine learning models for price prediction
- Unit testing for reliability

## Setup

This project uses UV as the package installer:

```bash
# Install UV
pip install uv

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install package in development mode
uv pip install -e ".[test]"
```

## Project Structure

```
├── app/                  # Application code
├── data/                 # Data directory
│   ├── input/           # Input datasets
│   └── output/          # Model outputs
├── notebook/            # Analysis notebooks
├── src/                 # Core implementation
│   ├── dataset.py         # Data processing
│   ├── feature.py      # Feature engineering
│   ├── inference.py    # Model inference
│   └── model.py        # Model implementation
├── tests/              # Unit tests
├── pyproject.toml      # Project configuration
├── setup.py            # Setup script
└── uv.lock             # Dependency lock file
```

## Development

The project uses:
- `uv` for dependency management
- `pytest` for testing

## Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_inference.py
```

## Features

- Data processing pipeline for insurance datasets
- Feature engineering for risk assessment
- Machine learning model for price prediction
- Unit tests for core functionality
