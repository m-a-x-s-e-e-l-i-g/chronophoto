@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.11 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 exit /b %errorlevel%
)

echo Preparing Chronophoto...
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 exit /b %errorlevel%

".venv\Scripts\python.exe" -m chronophoto %*
