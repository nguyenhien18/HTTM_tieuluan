@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "%~dp0stop_demo.py"
) else (
    python "%~dp0stop_demo.py"
)
if errorlevel 1 pause
