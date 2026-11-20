
@echo off
TITLE ADAPT-EDGE Research and Demonstration Platform
COLOR 0B

echo =====================================================================
echo           ADAPT-EDGE: Decision-Selective Computation
echo            for Resource-Bounded Edge AI Model Switching
echo =====================================================================
echo.
echo [1/3] Checking environment...

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    pause
    exit /b 1
)

echo [2/3] Checking server...
echo       Dashboard: http://127.0.0.1:8000
echo       API Docs:  http://127.0.0.1:8000/docs
echo.

echo [3/3] Starting FastAPI Server...
start "" http://127.0.0.1:8000

".venv\Scripts\python.exe" -m uvicorn server.app:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
