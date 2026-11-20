
@echo off
TITLE ADAPT-EDGE Research and Demonstration Platform
COLOR 0B

cd /d "%~dp0"

echo.
echo =====================================================================
echo       ADAPT-EDGE: Decision-Selective Computation
echo        for Resource-Bounded Edge AI Model Switching
echo =====================================================================
echo.

REM Check Python environment
echo [1/3] Checking Python environment...

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found.
    echo Please create the environment and install dependencies.
    pause
    exit /b 1
)

REM Check backend entry point
IF NOT EXIST "server\app.py" (
    echo [ERROR] FastAPI application not found: server\app.py
    pause
    exit /b 1
)

REM Check frontend production build
IF NOT EXIST "server\static_dist\index.html" (
    echo [WARNING] Production frontend build not found.
    echo The dashboard may not be available through FastAPI.
    echo.
)

echo [2/3] Checking server configuration...
echo.
echo Dashboard: http://127.0.0.1:8000
echo API Docs:  http://127.0.0.1:8000/docs
echo.

echo [3/3] Starting FastAPI server...
echo Press CTRL+C to stop the server.
echo.

start "" "http://127.0.0.1:8000"

".venv\Scripts\python.exe" -m uvicorn server.app:app --host 127.0.0.1 --port 8000

echo.
echo =====================================================================
echo ADAPT-EDGE server stopped.
echo =====================================================================
pause
