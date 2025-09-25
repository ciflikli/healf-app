#!/bin/bash

# Install dependencies from pyproject.toml
echo "Installing dependencies..."
python -m pip install -e .

# Start the application
echo "Starting the application..."
python main.py