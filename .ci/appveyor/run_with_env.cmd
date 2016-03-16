@echo off
:: To build extensions for 64 bit Python 3, we need to configure environment
:: variables to use the MSVC 2010 C++ compilers from GRMSDKX_EN_DVD.iso of:
:: MS Windows SDK for Windows 7 and .NET Framework 4
::
:: More details at:
:: https://github.com/cython/cython/wiki/64BitCythonExtensionsOnWindows
::
IF "%DISTUTILS_USE_SDK%"=="1" (
    IF "%PLATFORM%"=="x64" (
        ECHO Configuring environment to build with MSVC on a 64bit architecture
        ECHO Using Windows SDK v7.0
        "C:\Program Files\Microsoft SDKs\Windows\v7.0\Setup\WindowsSdkVer.exe" -q -version:v7.0
        CALL "C:\Program Files\Microsoft SDKs\Windows\v7.0\Bin\SetEnv.cmd" /x64 /release
        SET MSSdk=1
        REM Need the following to allow tox to see the SDK compiler
        SET TOX_TESTENV_PASSENV=DISTUTILS_USE_SDK MSSdk INCLUDE LIB
    ) ELSE (
        ECHO Using default MSVC build environment for 32 bit architecture
        ECHO Executing: %*
        call %* || EXIT 1
    )
) ELSE (
    ECHO Using default MSVC build environment
)

CALL %*
