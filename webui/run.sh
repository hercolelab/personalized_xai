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

dataset="diabetes"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|-d)
            dataset="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: ./webui/run.sh [--dataset <diabetes|lendingclub>]"
            exit 1
            ;;
    esac
done

streamlit run webui/streamlit_app.py -- --dataset "$dataset"
