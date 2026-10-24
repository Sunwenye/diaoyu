@echo off
chcp 65001 >nul
set "PY=C:\Users\sunwenye\AppData\Roaming\ennewClaw\runtimes\python-win\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0ring_detector.py" %*