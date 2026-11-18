@echo off
cd /d "%~dp0"
if "%INSIGHT_LLM_BACKEND%"=="" set INSIGHT_LLM_BACKEND=auto
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launch.py
) else (
    python launch.py
)
