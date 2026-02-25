@echo off
title BioEcho - AI Health Scanner
echo.
echo ============================================
echo   BioEcho v2 - AI Health Biomarker Scanner
echo ============================================
echo.

:: Prefer Python 3.12
set "PYTHON="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PYTHON=%LocalAppData%\Programs\Python\Python312\python.exe"
) else if exist "C:\Python312\python.exe" (
    set "PYTHON=C:\Python312\python.exe"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Install Python 3.12 from python.org
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

echo [INFO] Using: %PYTHON%

echo [1/3] Setting up virtual environment...
if not exist "bioecho_venv" (
    "%PYTHON%" -m venv bioecho_venv
    echo       Created bioecho_venv
) else (
    echo       Using existing bioecho_venv
)

echo [2/3] Installing dependencies...
bioecho_venv\Scripts\pip.exe install -r requirements.txt --quiet --disable-pip-version-check 2>nul
if errorlevel 1 (
    echo [WARN] Retrying individual packages...
    bioecho_venv\Scripts\pip.exe install customtkinter opencv-python numpy sounddevice scipy onnxruntime Pillow librosa --quiet 2>nul
)

echo [3/3] Launching BioEcho...
echo.
bioecho_venv\Scripts\python.exe bioecho_ui.py

if errorlevel 1 (
    echo.
    echo [ERROR] BioEcho exited with an error.
    pause
)
