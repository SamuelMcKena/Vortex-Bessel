@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "QT_API=pyside6"

if not exist ".venv\Scripts\python.exe" (
  echo Private environment is not installed yet.
  echo Running first-time setup...
  call INSTALL_AND_START.bat
  exit /b %errorlevel%
)

echo Starting Hexapod Lab with a visible debug console...
echo.
".venv\Scripts\python.exe" run_hexapod_lab.py
echo.
echo Process exited with code %errorlevel%.
pause
