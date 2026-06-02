@echo off
setlocal

call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
if errorlevel 1 exit /b 1

set "PGROOT=C:\Program Files\PostgreSQL\18"
set "SDKVER=10.0.26100.0"
set "SDKINC=C:\Program Files (x86)\Windows Kits\10\Include\%SDKVER%"
set "SDKLIB=C:\Program Files (x86)\Windows Kits\10\Lib\%SDKVER%"
set "MSVCLIB=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\lib\x64"
set "INCLUDE=%SDKINC%\ucrt;%SDKINC%\shared;%SDKINC%\um;%SDKINC%\winrt;%INCLUDE%"
set "LIB=%MSVCLIB%;%SDKLIB%\ucrt\x64;%SDKLIB%\um\x64;%LIB%"
set "PATH=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64;%PATH%"

cd /d C:\tmp\pgvector_build_082
if errorlevel 1 exit /b 1

"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\nmake.exe" /F Makefile.win clean
if errorlevel 1 exit /b 1

"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\nmake.exe" /F Makefile.win
if errorlevel 1 exit /b 1

"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\nmake.exe" /F Makefile.win install
exit /b %ERRORLEVEL%
