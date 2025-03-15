@echo off
title ChatGLM-CPU-Trainer 一键训练
cd /d "%~dp0.."

echo [信息] 启动ChatGLM训练...

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
    echo [信息] 正在构建镜像...
    docker build -t chatglm-cpu-trainer .
    if %ERRORLEVEL% NEQ 0 (
        echo [错误] 构建镜像失败！请检查Dockerfile
        pause
        goto :EOF
    )
)

echo [信息] 开始训练过程，这可能需要几分钟到几小时不等...
echo [信息] 首次运行时将自动下载模型(大约3-6GB)，请耐心等待
echo [信息] 训练进度分为三个阶段:
echo       1. 模型下载: 首次运行需要，下载ChatGLM模型
echo       2. 数据准备: 处理训练数据集，分词和编码
echo       3. 模型训练: 使用LoRA进行实际训练
echo.
echo [提示] 如果模型下载速度较慢，可以在.env文件中修改HF_ENDPOINT镜像站点

echo [信息] 开始训练，这可能需要较长时间...
docker-compose run train

if %ERRORLEVEL% NEQ 0 (
    echo [错误] 训练过程中出现错误
) else (
    echo [成功] 训练完成！
)
pause