@echo off
cd /d "%~dp0.."
python -m pip install -r research_extension\requirements.txt
if errorlevel 1 exit /b 1
python -m research_extension.run --backend none
if errorlevel 1 exit /b 1
python -m pytest -q research_extension\tests
