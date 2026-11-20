@echo off
TITLE ADAPT-EDGE Development Mode (Vite + FastAPI)
COLOR 0A

echo Starting ADAPT-EDGE in Hot-Reload Development Mode...
echo Backend: http://127.0.0.1:8000
echo Frontend: http://localhost:3000
echo.

start cmd /k "color 0B && title ADAPT-EDGE Backend && .venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload"
cd frontend
npm run dev

