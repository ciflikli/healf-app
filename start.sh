#!/bin/bash

# Install dependencies from pyproject.toml
echo "Installing dependencies..."
pip install -e .

# Start the application
echo "Starting the application..."
python main.py