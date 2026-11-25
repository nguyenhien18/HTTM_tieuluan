@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 scripts\check_package.py
if errorlevel 1 (
 echo Tep ban giao da thay doi hoac thieu. Doi chieu manifest truoc khi tiep tuc.
 pause
 exit /b 1
)
python -X utf8 scripts\verify_submission.py --with-llm
if errorlevel 1 (
 echo Co buoc kiem tra chua dat. Xem loi o tren.
) else (
 echo Da kiem tra xong. Ket qua: runs\verification\submission.json
)
pause
