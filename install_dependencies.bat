@echo off
setlocal
title ISL Dependencies Installer

echo ========================================================
echo   Installing Dependencies for ISL Interpreter...
echo ========================================================
echo.

set "PY_EXE="

if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto :install_now
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_EXE=python"
    goto :install_now
)

echo [ERROR] Python not found!
pause
exit /b 1

:install_now
echo Using Python at: %PY_EXE%
echo Installing packages from requirements.txt...
"%PY_EXE%" -m pip install -r requirements.txt
echo.
echo ========================================================
echo   Installation finished! You can now run:
echo   - run.bat (for Web Interpreter)
echo   - desktop.bat (for Desktop OpenCV Interpreter)
echo ========================================================
pause

