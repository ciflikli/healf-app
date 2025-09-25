#!/bin/bash

# Create and activate virtual environment
echo "Setting up virtual environment..."
python -m venv venv
source venv/bin/activate

# Ensure pip is available
echo "Ensuring pip is available..."
python -m ensurepip --upgrade

# Install dependencies from pyproject.toml
echo "Installing dependencies..."
python -m pip install -e .

# Start the application
echo "Starting the application..."
python main.py