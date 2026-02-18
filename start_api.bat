@echo off
echo Starting AI Academic Advisor API Server...
echo.
echo This will keep the API running continuously.
echo Press Ctrl+C to stop the server.
echo.

REM Set environment variables
set AI_ADVISOR_MODEL_NAME=auto

REM Start API server
python api_app.py

pause
