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
- `pyproject.toml`: Project configuration and dependencies
- `uv.lock`: Locked dependencies for reproducible builds

## Development

The project uses modern Python tooling:
- `uv` for dependency management
- `pytest` for testing
- `pydantic` for data validation
- `langchain` for LLM integration

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
``` 