@echo off
setlocal
set "LLM_MODEL=gemma4:e4b"
call "%~dp03-run-ai-play.bat" %*
exit /b %ERRORLEVEL%
