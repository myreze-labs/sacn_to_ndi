@echo off
REM =========================================================
REM sACN to NDI Bridge - Windows Launcher
REM =========================================================
REM This script updates the code, installs dependencies,
REM and launches the bridge with web UI.
REM =========================================================

echo.
echo ========================================
echo   sACN to NDI Bridge - Starting...
echo ========================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

REM ── Git Pull ──
echo [1/3] Updating code from Git...
git pull
if errorlevel 1 (
    echo.
    echo WARNING: Git pull failed. Continuing anyway...
    echo.
    timeout /t 2 >nul
)

REM ── Install/Update Dependencies ──
echo.
echo [2/3] Installing/updating dependencies...

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Found virtual environment, activating...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
    echo TIP: Create one with: python -m venv venv
    echo.
)

python -m pip install --upgrade pip -q
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Please check your Python installation and try again.
    pause
    exit /b 1
)

REM ── Launch Bridge ──
echo.
echo [3/3] Launching sACN to NDI Bridge...
echo.
echo Web UI will be available at: http://localhost:8080
echo Press Ctrl+C to stop the bridge
echo.
echo ========================================
echo.

REM Launch with web UI enabled on port 8080
REM Adjust the parameters below as needed:
REM   -u 1          = Universe 1 (change to -u 1 2 3 for multiple)
REM   --web         = Enable web UI
REM   --web-port    = Port for web UI (default 8080)
REM   --mock        = Use mock NDI (remove for real NDI SDK)

python main.py -u 1 --web --web-port 8080

REM If the script exits, pause so user can see any errors
echo.
echo.
echo Bridge stopped.
pause
