@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%scripts\start_app.ps1" notebook %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Open Notebook failed with exit code %EXIT_CODE%.
    echo Check the error above, then press any key to close this window.
    pause >nul
)
exit /b %EXIT_CODE%
