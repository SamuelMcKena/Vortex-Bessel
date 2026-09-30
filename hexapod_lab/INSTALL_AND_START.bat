@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo   Hexapod + Laser Lab - first-time setup
echo ==============================================
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" was not found.
  echo Install 64-bit Python 3.11 or 3.12 and enable the Python launcher,
  echo then run this file again.
  pause
  exit /b 1
)

set "PYVER="
py -3.12 -c "import sys" >nul 2>nul && set "PYVER=-3.12"
if not defined PYVER py -3.11 -c "import sys" >nul 2>nul && set "PYVER=-3.11"
if not defined PYVER set "PYVER=-3"

if not exist ".venv\Scripts\python.exe" (
  echo Creating private Python environment using %PYVER% ...
  py %PYVER% -m venv .venv
  if errorlevel 1 goto :failed
)

echo Installing/updating GUI and exact-CAD dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements-cad.txt
if errorlevel 1 goto :failed

echo Running a quick source check...
".venv\Scripts\python.exe" -m compileall -q hexapod_lab run_hexapod_lab.py
if errorlevel 1 goto :failed

echo.
echo Starting Hexapod + Laser Lab in its configured startup mode...
start "" ".venv\Scripts\pythonw.exe" run_hexapod_lab.py
exit /b 0

:failed
echo.
echo Setup failed. Read the messages above, then send the full window/output
echo if you want me to patch the installer for this PC.
pause
exit /b 1
