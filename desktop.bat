@echo off
setlocal
title ISL Desktop Interpreter (OpenCV Window)

echo ========================================================
echo   Indian Sign Language (ISL) Desktop Interpreter
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
echo [2/2] Launching Standalone Desktop Window...
echo.
"%PY_EXE%" desktop_interpreter.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================================
    echo   An error occurred!
    echo   To install/fix all required dependencies, run:
    echo   "%PY_EXE%" -m pip install -r requirements.txt
    echo ========================================================
    pause
)

