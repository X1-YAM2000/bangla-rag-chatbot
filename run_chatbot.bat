@echo off
REM run_chatbot.bat  -  Run the Bangla RAG Chatbot directly
REM Works with system Python (no venv required)
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python src\bangla_rag_chatbot.py
