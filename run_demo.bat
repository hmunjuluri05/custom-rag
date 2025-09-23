@echo off
echo 🎭 RAG System Demo Mode
echo ============================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo 🐍 Python found
echo 🚀 Starting demo mode...
echo.

REM Run the demo script
python run_demo.py

pause