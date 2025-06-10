from setuptools import setup, find_packages

setup(
    name="underwriting-assessor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-anthropic",
        "python-dotenv",
        "pydantic",
        "anthropic",
        "pypdf",
    ],
) 