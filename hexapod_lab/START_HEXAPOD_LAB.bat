@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Private environment is not installed yet.
  echo Running first-time setup...
  call INSTALL_AND_START.bat
  exit /b %errorlevel%
)

if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" run_hexapod_lab.py
  exit /b 0
)

".venv\Scripts\python.exe" run_hexapod_lab.py
if errorlevel 1 (
  echo.
  echo Hexapod Lab exited with an error.
  echo Check hexapod_lab_crash.log in this folder.
  pause
)
