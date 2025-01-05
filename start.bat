@echo off
setlocal

REM Get the directory of the script
set SCRIPT_DIR=%~dp0

REM Check if the virtual environment exists
if not exist "%SCRIPT_DIR%exllamav2_env\Scripts\activate.bat" (
    echo ERROR: The virtual environment does not exist.
    echo Please run the windows_install.bat script first to set up the environment.
    pause
    goto :end
)

REM Activate the virtual environment
call "%SCRIPT_DIR%exllamav2_env\Scripts\activate.bat"

REM Run the script
python "%SCRIPT_DIR%autoconvert.py"

REM Deactivate the virtual environment
deactivate

:end
endlocal
