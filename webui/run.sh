#!/bin/bash

echo "Starting Personalized XAI Web UI..."
echo "======================================="
echo ""

if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed."
    echo "Please install it with: pip install streamlit"
    exit 1
fi

cd "$(dirname "$0")/.."

streamlit run webui/streamlit_app.py
