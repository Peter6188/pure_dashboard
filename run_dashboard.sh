#!/bin/bash

# Script to run the Bank Marketing Dashboard
# This ensures the virtual environment is activated and packages are available

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting Bank Marketing Dashboard..."
echo "Dashboard will be available at: http://127.0.0.1:8051"
echo "Press Ctrl+C to stop the dashboard"
echo ""

python dashboard_simple.py
