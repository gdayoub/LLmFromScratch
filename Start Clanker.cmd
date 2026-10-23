@echo off
setlocal
title Clanker
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Clanker needs the project Python environment
    echo See README.md for setup instructions
    pause
    popd
    exit /b 1
)
if not exist "runs\animals\best.pt" (
    echo Clanker cannot find runs\animals\best.pt
    echo This repository does not include trained checkpoints
    echo See README.md for training instructions
    pause
    popd
    exit /b 1
)
".venv\Scripts\python.exe" -B "sample_subword.py" %*
set "clanker_exit=%errorlevel%"
if not "%clanker_exit%"=="0" pause
popd
exit /b %clanker_exit%
