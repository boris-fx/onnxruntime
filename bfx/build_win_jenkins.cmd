@echo off

cd %WORKSPACE%

CALL "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"

REM jenkins adds a path to jre that is wrapped by double quotes to PATH and __VSCMD_PREINIT_PATH. remove the double quotes to make msvc & nvcc happy!
SET PATH=%PATH:"=%
SET __VSCMD_PREINIT_PATH=%__VSCMD_PREINIT_PATH:"=%

set ORT_VERSION=1.20.2

REM is this really the best way to do this in CMD???
>temp.txt ( git rev-parse --short HEAD )
set /p GIT_HASH=<temp.txt
del temp.txt
set GIT_HASH=%GIT_HASH:~0,7%

set DIST_NAME=libonnxruntime-%ORT_VERSION%_win_cu118-dml-1.15.2_%GIT_HASH%_%BUILD_NUMBER%

Powershell.exe -File bfx/build_win.ps1 %DIST_NAME%

rclone copy build\%DIST_NAME%.zip "mescola:Boris FX/Engineering/BinaryArtifacts"

EXIT 0
