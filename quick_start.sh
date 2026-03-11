#!/usr/bin/env bash
# quick_start.sh  -  Bangla RAG Chatbot launcher (Linux / Mac / WSL)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONIOENCODING=utf-8

echo ""
echo "============================================================"
echo "  Bangla RAG Chatbot - Quick Start"
echo "============================================================"
echo ""

# Install from project root where requirements.txt lives
echo "Installing dependencies..."
python3 -m pip install -q -r requirements.txt
echo "Dependencies installed."
echo ""

echo "============================================================"
echo "  Select an option:"
echo "  1. Interactive Chatbot"
echo "  2. Run Tests"
echo "  3. Run Demo"
echo "  4. Start FastAPI Server (port 8000)"
echo "============================================================"
echo ""
read -r -p "Enter choice (1-4): " choice

case "$choice" in
    1)
        echo "Starting chatbot..."
        python3 src/bangla_rag_chatbot.py
        ;;
    2)
        echo "Running tests..."
        python3 src/test_chatbot.py
        ;;
    3)
        echo "Running demo..."
        python3 src/demo_chatbot.py
        ;;
    4)
        echo "Starting FastAPI server at http://localhost:8000"
        echo "Press Ctrl+C to stop."
        python3 -m uvicorn src.api:app --reload --port 8000
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac
