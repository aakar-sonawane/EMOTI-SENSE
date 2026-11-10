@echo off
rem EmotiSense launcher: double-click this file. Ollama is started automatically by the server.
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" (call "venv\Scripts\activate.bat") else (echo [!] venv folder not found. Create it first, see README.md & pause & exit /b 1)
start "" cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8000"
python -m uvicorn server:app --app-dir backend --host 0.0.0.0 --port 8000
pause
