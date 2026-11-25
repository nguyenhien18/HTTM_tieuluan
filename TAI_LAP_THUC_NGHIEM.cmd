@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 scripts\reproduce.py --train-tabular
if errorlevel 1 echo Co buoc chua dat. Xem loi o tren.
pause
