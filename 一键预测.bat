@echo off
title ChatGLM-CPU-Trainer 一键预测
cd /d "%~dp0"

:: 检查Docker是否运行
docker info >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] Docker引擎未运行，请先启动Docker！
    pause
    goto :EOF
)

:: 检查镜像是否存在
docker image inspect chatglm-cpu-trainer >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未找到chatglm-cpu-trainer镜像！
    echo [信息] 您需要先构建镜像或运行训练
    pause
    goto :EOF
)

:: 检查模型是否存在
if not exist "models\chatglm-lora" (
    echo [警告] 未找到训练好的模型，将使用基础模型预测
    echo [信息] 建议先运行训练，或确保模型路径正确
)

set /p PROMPT=请输入提示词:

echo [信息] 开始生成回复...
setlocal EnableDelayedExpansion
docker-compose run -e "PROMPT=!PROMPT!" predict

if %ERRORLEVEL% NEQ 0 (
    echo [错误] 预测过程中出现错误
) else (
    echo [成功] 预测完成！

    :: 尝试显示结果
    echo [信息] 预测结果:
    echo ==================================
    type "data\output\prediction.txt" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [警告] 无法显示预测结果，请手动查看输出文件
    )
    echo ==================================
)

pause