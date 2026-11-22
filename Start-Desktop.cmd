@echo off
setlocal
cd /d "%~dp0app"
if not exist ".venv\Scripts\python.exe" (
    call setup.cmd
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m mahjong_helper %*
if errorlevel 1 (
    pause
    exit /b 1
)
