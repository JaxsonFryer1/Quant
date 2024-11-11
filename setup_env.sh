#!/bin/bash

pip install --upgrade pip


# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it to proceed."
    exit 1
fi

# Create a virtual environment in the current directory
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found!"
    deactivate
    exit 1
fi

# Install packages from requirements.txt
pip install -r requirements.txt

# Confirm installation
echo "Packages installed in the virtual environment located at $(pwd)/venv"

echo "Now Select the correct python interpreter using cmd + shift + p and select the venv"