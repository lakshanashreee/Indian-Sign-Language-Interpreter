@echo off
setlocal
title ISL Interpreter Launcher

echo ========================================================
echo   Indian Sign Language (ISL) AI Interpreter
echo ========================================================
echo.

set "PY_EXE="

if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto :start_app
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_EXE=python"
    goto :start_app
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_EXE=py"
    goto :start_app
)

echo [ERROR] Python was not found on your system!
pause
exit /b 1

:start_app
echo [1/2] Verifying Python interpreter: %PY_EXE%
echo [2/2] Launching Web Interpreter...
echo.
"%PY_EXE%" run_interpreter.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================================
    echo   An error occurred during startup!
    echo   To install/fix all required dependencies, run:
    echo   "%PY_EXE%" -m pip install -r requirements.txt
    echo ========================================================
    pause
)

