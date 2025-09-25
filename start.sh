#!/bin/bash

# This is a Python project - no npm needed

# Use Python 3.11 directly from Nix
echo "Using Python 3.11 from Nix environment..."
python3 --version

# Install dependencies directly without virtual environment
echo "Installing dependencies..."
pip install --user fastapi uvicorn numpy openai polars polars-ols pyarrow python-multipart scipy statsmodels

# Start the FastAPI application with uvicorn
echo "Starting FastAPI server on port 5000..."
uvicorn main:app --host 0.0.0.0 --port 5000