# Underwriting Assessor

A machine learning-powered tool for predicting insurance pricing to assist underwriters in risk assessment and premium calculation.

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
