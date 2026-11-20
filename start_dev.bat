
@echo off
TITLE ADAPT-EDGE Development Mode
COLOR 0A

cd /d "%~dp0"

echo ============================================================
echo        ADAPT-EDGE - DEVELOPMENT MODE
echo ============================================================
echo.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:3000
echo API Docs: http://127.0.0.1:8000/docs
echo.

REM Check Python environment
IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found.
    pause
    exit /b 1
)

REM Check frontend dependencies
IF NOT EXIST "frontend\node_modules" (
    echo [ERROR] Frontend dependencies are missing.
    echo Run: cd frontend ^&^& npm install
    pause
    exit /b 1
)

REM Start FastAPI backend
echo [1/2] Starting FastAPI backend...

start "ADAPT-EDGE Backend" cmd /k "cd /d "%~dp0" && color 0B && title ADAPT-EDGE Backend && .venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload"

REM Start React frontend
echo [2/2] Starting React frontend...
echo.

cd /d "%~dp0frontend"
npm run dev

echo.
echo Frontend development server stopped.
pause
