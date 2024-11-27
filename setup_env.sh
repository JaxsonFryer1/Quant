#!/bin/bash

# Upgrade system-wide pip to avoid potential issues
echo "Upgrading system-wide pip..."
pip install --upgrade pip

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it to proceed."
    exit 1
fi

# Create a virtual environment
echo "Creating a virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Upgrade pip in the virtual environment
echo "Upgrading pip in the virtual environment..."
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found! Create one or place it in the current directory."
    deactivate
    exit 1
fi

# Install packages from requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

# Confirm successful setup
echo "Packages installed successfully in the virtual environment located at $(pwd)/venv"
echo "Use the correct Python interpreter in your IDE (e.g., cmd + shift + p in VSCode)."
