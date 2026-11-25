@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
 ".venv\Scripts\python.exe" scripts\verify_demo.py --with-llm
) else (
 python scripts\verify_demo.py --with-llm
)
pause
