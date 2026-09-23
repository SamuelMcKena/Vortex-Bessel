@echo off
setlocal
cd /d "%~dp0"
python run_hexapod_lab.py
if errorlevel 1 (
  echo.
  echo Hexapod Lab exited with an error.
  pause
)
endlocal
