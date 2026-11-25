@echo off
cd /d "%~dp0"
title shu-insight local server

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Please install first, see README section 1.
    pause
    exit /b 1
)

set "SHU_NOTICE_DATABASE_PATH="
if not exist "data\shu_insight.sqlite3" goto run
".venv\Scripts\python.exe" -c "import sys;open(sys.argv[1],'a').close()" "data\.write-probe" >nul 2>&1
if not errorlevel 1 goto probe_ok
del "data\.write-probe" >nul 2>&1
if not exist "shu_insight.sqlite3" copy /y "data\shu_insight.sqlite3" "shu_insight.sqlite3" >nul 2>&1
set "SHU_NOTICE_DATABASE_PATH=%~dp0shu_insight.sqlite3"
echo [note] data folder is read-only in this session; using root copy of the database.
goto run

:probe_ok
del "data\.write-probe" >nul 2>&1

:run
if defined SHU_NOTICE_DATABASE_PATH (echo [db] %SHU_NOTICE_DATABASE_PATH%) else (echo [db] %cd%\data\shu_insight.sqlite3)
echo Starting shu-insight, please wait about 6 seconds...
echo Browser will open http://127.0.0.1:8501
echo Keep this black window open while using the app.
echo.

start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep 6; Start-Process 'http://127.0.0.1:8501'"

".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false

echo.
echo Server stopped. Press any key to close this window.
pause >nul
