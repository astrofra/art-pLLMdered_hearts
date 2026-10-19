@echo off
setlocal
set "LLM_MODEL=qwen3.5:4b"
set "LLM_THINK=0"
call "%~dp03-run-ai-play.bat" %*
exit /b %ERRORLEVEL%
