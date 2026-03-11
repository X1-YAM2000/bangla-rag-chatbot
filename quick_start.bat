@echo off
chcp 65001 >nul
setlocal

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"
set PYTHONIOENCODING=utf-8

echo.
echo ============================================================
echo   Bangla RAG Chatbot - Quick Start
echo ============================================================
echo.

REM Install dependencies from project root (requirements.txt is here)
echo Installing dependencies...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)
echo Dependencies installed.
echo.

echo ============================================================
echo   Select an option:
echo   1. Interactive Chatbot
echo   2. Run Tests
echo   3. Run Demo
echo   4. Start FastAPI Server (port 8000)
echo ============================================================
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo Starting chatbot...
    python src\bangla_rag_chatbot.py
) else if "%choice%"=="2" (
    echo Running tests...
    python src\test_chatbot.py
) else if "%choice%"=="3" (
    echo Running demo...
    python src\demo_chatbot.py
) else if "%choice%"=="4" (
    echo Starting FastAPI server at http://localhost:8000
    echo Press Ctrl+C to stop.
    python -m uvicorn src.api:app --reload --port 8000
) else (
    echo Invalid choice.
    exit /b 1
)
