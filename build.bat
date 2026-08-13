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

echo Installing build tools...
".venv\Scripts\python.exe" -m pip install -e ".[build]"
if errorlevel 1 exit /b %errorlevel%

echo Building Chronophoto...
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\build_nvidia_runtime.ps1"
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean chronophoto.spec
if errorlevel 1 exit /b %errorlevel%

"dist\Chronophoto\Chronophoto.exe" --version
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build ready: dist\Chronophoto\Chronophoto.exe
