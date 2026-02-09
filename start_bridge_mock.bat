@echo off
REM =========================================================
REM sACN to NDI Bridge - Quick Start (Mock Mode)
REM =========================================================
REM This script launches the bridge in MOCK mode for testing
REM without requiring the NDI SDK to be installed.
REM =========================================================

echo.
echo ========================================
echo   sACN to NDI Bridge (Mock Mode)
echo ========================================
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo Starting bridge in MOCK mode (no NDI SDK required)...
echo Web UI: http://localhost:8080
echo.

python main.py -u 1 --web --web-port 8080 --mock

echo.
echo Bridge stopped.
pause
