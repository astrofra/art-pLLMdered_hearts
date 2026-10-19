@echo off
setlocal
set "LLM_MODEL=gemma4:e4b"
set "LLM_THINK=0"
call "%~dp03-run-ai-play.bat" %*
exit /b %ERRORLEVEL%
