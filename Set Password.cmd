@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe set_password.py
echo.
pause
