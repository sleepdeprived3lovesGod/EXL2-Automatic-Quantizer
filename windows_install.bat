@echo off
setlocal

REM Get the directory of the script
set SCRIPT_DIR=%~dp0

REM Function to check if Python is installed
:check_python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed.
    echo Please download and install Python from https://www.python.org/downloads/.
    echo Make sure to check the box that says "Add Python to PATH" before clicking "Install Now".
    goto :end
)

REM Function to check if Git is installed
:check_git
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed.
    echo Please download and install Git from https://git-scm.com/download/win.
    echo You can use the default settings during installation.
    goto :end
)

REM Function to check if CUDA Toolkit is installed
:check_cuda
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CUDA Toolkit is not installed.
    echo Please download and install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads.
    echo Make sure to install the necessary components, including the CUDA driver, CUDA toolkit, and cuDNN library.
    goto :end
)

REM Create a virtual environment in the script directory
python -m venv "%SCRIPT_DIR%exllamav2_env"

REM Activate the virtual environment
call "%SCRIPT_DIR%exllamav2_env\Scripts\activate.bat"

REM Install required Python packages
pip install configparser

REM Check if tkinter is available
python -c "import tkinter" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: tkinter is not available. Please ensure that Python was installed with tkinter support.
    goto :end
)

REM Clone the Exllamav2 repository
git clone https://github.com/turboderp/exllamav2 "%SCRIPT_DIR%exllamav2"
cd "%SCRIPT_DIR%exllamav2"
pip install -r requirements.txt
pip install .

REM Deactivate the virtual environment
deactivate

echo Installation complete. You can now run the script using the following commands:
echo 1. Open Command Prompt.
echo 2. Navigate to the directory where you ran the install script.
echo 3. Activate the virtual environment:
echo    call "%SCRIPT_DIR%exllamav2_env\Scripts\activate.bat"
echo 4. Run the script:
echo    python "%SCRIPT_DIR%autoconvert.py"
echo 5. Deactivate the virtual environment when done:
echo    deactivate

:end
endlocal
