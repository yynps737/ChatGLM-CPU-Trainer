@echo off
title ChatGLM-CPU-Trainer 一键训练
cd /d "%~dp0.."

echo 启动ChatGLM训练...
docker-compose run train

echo 训练完成！
pause