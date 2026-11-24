@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 py -3.12 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r config\requirements-tested.txt
if errorlevel 1 goto failed
echo Setup complete.
if /i not "%~1"=="--quiet" pause
exit /b 0
:failed
echo Setup failed. See README.md for prerequisites.
if /i not "%~1"=="--quiet" pause
exit /b 1
