#!/bin/bash

# Remove any existing virtual environment to ensure a clean setup
echo "Cleaning up any existing virtual environment..."
rm -rf venv

# Create and activate virtual environment
echo "Setting up virtual environment..."
python -m venv venv
source venv/bin/activate

# Capture the virtual environment's Python executable
VENV_PYTHON=$(which python)

# Ensure pip is available
echo "Ensuring pip is available..."
$VENV_PYTHON -m ensurepip --upgrade
$VENV_PYTHON -m pip install --upgrade pip

# Install fundamental packages for dependency resolution
echo "Installing setuptools and wheel..."
$VENV_PYTHON -m pip install --upgrade setuptools wheel

# Install dependencies from pyproject.toml
echo "Installing dependencies..."
$VENV_PYTHON -m pip install -e .

# Start the application
echo "Starting the application..."
$VENV_PYTHON main.py