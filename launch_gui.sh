#!/bin/bash
# Lada GUI Launcher Script
# This script launches the Lada GUI using the correct Python environment

# Change to the lada directory
cd "/Users/ry/Downloads/lada-main mac"

# Add Homebrew to PATH
eval "$(/opt/homebrew/bin/brew shellenv)"

# Activate the GUI virtual environment
source .venv-gui/bin/activate

# Launch the GUI
lada
