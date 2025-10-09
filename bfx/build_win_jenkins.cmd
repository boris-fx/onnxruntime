@echo off

cd %WORKSPACE%

SET VCVARS_X86_64=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat
SET VCVARS_ARM64=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsamd64_arm64.bat

set ORT_VERSION=1.20.2

REM is this really the best way to do this in CMD???
>temp.txt ( git rev-parse --short HEAD )
set /p GIT_HASH=<temp.txt
del temp.txt
set GIT_HASH=%GIT_HASH:~0,7%
set BUILD_DATE=%date:~10,4%-%date:~4,2%-%date:~7,2%
set BUILD_ID=%BUILD_DATE%_%GIT_HASH%_%BUILD_NUMBER%

set DIST_NAME=libonnxruntime-%ORT_VERSION%_win_cu128-dml-1.15.4_%BUILD_ID%

Powershell.exe -File bfx/build_win.ps1 %DIST_NAME% || goto :error

rclone copy build\%DIST_NAME%.zip "mescola:Boris FX/Engineering/BinaryArtifacts"

:;
exit /b 0

:error
echo ERROR: %errorlevel%
exit /b %errorlevel%
