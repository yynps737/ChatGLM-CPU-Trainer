@echo off
title ChatGLM-CPU-Trainer 一键预测
cd /d "%~dp0.."

set /p PROMPT=请输入提示词:
setlocal EnableDelayedExpansion
docker-compose run -e "PROMPT=!PROMPT!" predict

echo 预测完成！
pause