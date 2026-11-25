@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
title HTTM Demo Studio
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "%~dp0launch.py"
) else (
    python "%~dp0launch.py"
)
if errorlevel 1 (
    echo.
    echo Khong khoi dong duoc. Xem thong bao phia tren.
    echo Cai thu vien bang: python -m pip install -r requirements.txt
    pause
)
