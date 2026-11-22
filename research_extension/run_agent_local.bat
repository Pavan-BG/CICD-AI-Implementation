@echo off
cd /d "%~dp0.."
python -m research_extension.run --backend ollama --model qwen2.5:0.5b --suggest
