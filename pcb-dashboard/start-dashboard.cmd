@echo off
setlocal

where node.exe >nul 2>&1 && where npm.cmd >nul 2>&1 && goto run

for /d %%I in ("%LOCALAPPDATA%\JetBrains\PyCharm*") do (
    for /d %%R in ("%%~fI\acp-agents\.runtimes\node\*") do (
        if exist "%%~fR\node.exe" if exist "%%~fR\npm.cmd" set "PCB_NODE_RUNTIME=%%~fR"
    )
)
if not defined PCB_NODE_RUNTIME (
    echo Node.js and npm were not found. Install Node.js LTS, then reopen your terminal.
    exit /b 1
)
set "PATH=%PCB_NODE_RUNTIME%;%PATH%"

:run
pushd "%~dp0" || exit /b 1
if not exist "node_modules\.bin\vite.cmd" (
    call npm.cmd ci
    if errorlevel 1 goto failed
)
call npm.cmd run dev %*
set "PCB_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %PCB_EXIT_CODE%

:failed
popd
exit /b 1
