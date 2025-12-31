@echo off
setlocal

REM One-click launcher for MCQGen Streamlit UI.
REM Creates a local venv, installs deps, then runs Streamlit.

set "VENV_DIR=.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Creating virtual environment...
  python -m venv "%VENV_DIR%"
)

echo Using: %PYTHON_EXE%
echo Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

echo Installing dependencies...
"%PYTHON_EXE%" -m pip install -e .

echo Starting Streamlit...
"%PYTHON_EXE%" -m streamlit run streamlit_app.py
