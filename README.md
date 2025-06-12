# Underwriting Risk Assessment

This project focuses on underwriting risk assessment using machine learning techniques.

## Setup

This project uses UV as the package installer. To set up the project:

1. Install UV if you haven't already:
```bash
pip install uv
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install the package in development mode with test dependencies
uv pip install -e ".[test]"
```

## Project Structure

- `src/`: Source code directory containing the main application code
- `tests/`: Test files directory
- `data/`: Directory for input and output data
- `train/`: Directory containing experiment tracking implementations
  - `exp.py`: MLflow-based experiment tracking
  - `exp2.py`: LangChain-based experiment tracking
- `pyproject.toml`: Project configuration and dependencies
- `uv.lock`: Locked dependencies for reproducible builds

## Development

The project uses modern Python tooling:
- `uv` for dependency management
- `pytest` for testing
- `pydantic` for data validation
- `langchain` for LLM integration
- `mlflow` for experiment tracking

## Experiment Tracking

The project provides two options for tracking LLM experiments:

### MLflow-based Tracking (exp.py)
```python
from train.exp import run_experiment

run_experiment(
    experiment_name="my_experiment",
    model_name="gpt-3.5-turbo",
    prompt="Your prompt here",
    input_data=[{"text": "Sample input"}],
    temperature=0.8,
    max_tokens=150
)
```

### LangChain-based Tracking (exp2.py)
```python
from train.exp2 import run_experiment

run_experiment(
    experiment_name="my_experiment",
    model_name="gpt-3.5-turbo",
    system_prompt="You are a helpful AI assistant.",
    user_prompt="Your question here",
    temperature=0.8,
    max_tokens=500
)
```

## Running Tests

To run the tests:
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src

# Run specific test file
pytest tests/test_inference.py

# Run specific test function
pytest tests/test_inference.py::test_process_document
```

## Environment Variables

Create a `.env` file in the project root with:
```
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Features

- **Multiple Experiment Tracking Options**:
  - MLflow-based tracking (`exp.py`)
  - LangChain-based tracking (`exp2.py`)
  - System metrics monitoring
  - Prompt and output logging
  - Performance metrics tracking

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd underwriting-assessor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Project Structure

```
underwriting-assessor/
├── train/
│   ├── exp.py          # MLflow-based experiment tracking
│   └── exp2.py         # LangChain-based experiment tracking
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Features Comparison

### MLflow Implementation (exp.py)
- Comprehensive experiment tracking
- Artifact management
- Parameter logging
- System metrics tracking
- Structured output storage

### LangChain Implementation (exp2.py)
- Native LangChain integration
- File-based experiment tracking
- System metrics through callbacks
- Prompt and output versioning
- Simplified setup

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 