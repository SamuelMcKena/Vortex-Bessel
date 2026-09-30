@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" run_hexapod_lab.py
  exit /b 0
)
python run_hexapod_lab.py
if errorlevel 1 (
  echo.
  echo Hexapod Lab exited with an error.
  pause
)
endlocal
