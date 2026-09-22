@echo off
setlocal
cd /d "%~dp0"
python tools	est_beamage_pipe.py
pause
