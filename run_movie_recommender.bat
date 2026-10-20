@echo off
setlocal

set "SCRIPT=%~dp0movie_recommender.py"

if not exist "%SCRIPT%" (
    echo ERROR: movie_recommender.py not found at:
    echo "%SCRIPT%"
    pause
    exit /b 1
)

cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%SCRIPT%"
) else (
    py -3 "%SCRIPT%"
)

if errorlevel 1 pause
