@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python was not found.
  echo Install 64-bit Python 3.11 or 3.12 from https://www.python.org/downloads/
  echo and enable "Add Python to PATH", then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the private Python environment...
  py -3 -m venv .venv
  if errorlevel 1 goto :failed
)

echo Installing or updating GUI and exact-CAD dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements-cad.txt
if errorlevel 1 goto :failed

echo Starting Hexapod + Laser Lab...
start "" ".venv\Scripts\pythonw.exe" run_hexapod_lab.py
exit /b 0

:failed
echo.
echo Installation failed. Check the messages above and the internet connection.
pause
exit /b 1
