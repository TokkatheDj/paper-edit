@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First run: setting up. This takes a couple of minutes.
  python -m venv .venv || goto :fail
  .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
)

echo.
echo   Starting Paper Edit. The addresses you can open it on are listed below.
echo   Leave this window open - closing it stops the editor.

.venv\Scripts\python.exe server.py
goto :eof

:fail
echo.
echo   Setup failed. Check that Python is installed and on your PATH.
pause
