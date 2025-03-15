@echo off
setlocal EnableDelayedExpansion
title ChatGLM-CPU-Trainer 自动配置工具

echo [信息] 检查Docker环境...
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] Docker未安装! 请先安装Docker: https://docs.docker.com/get-docker/
    goto :EOF
)

where docker-compose >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [警告] 未找到docker-compose命令，尝试使用docker compose...
    docker compose version >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [错误] Docker Compose未安装! 请安装Docker Compose
        goto :EOF
    )
    set use_new_compose=true
) else (
    set use_new_compose=false
)

:: 获取系统内存信息
for /f "tokens=4" %%a in ('systeminfo ^| findstr "Total Physical Memory"') do (
    set mem_str=%%a
    set mem_str=!mem_str:,=!
)
set /a total_memory_mb=mem_str
set /a total_memory_gb=total_memory_mb/1024

:: 根据内存大小设置配置
echo [信息] 检测到系统内存: %total_memory_gb%GB
if %total_memory_gb% LSS 6 (
    set memory_config=4gb
    echo [信息] 将使用4GB内存优化配置
) else if %total_memory_gb% LSS 12 (
    set memory_config=8gb
    echo [信息] 将使用8GB内存优化配置
) else if %total_memory_gb% LSS 24 (
    set memory_config=16gb
    echo [信息] 将使用16GB内存优化配置
) else (
    set memory_config=32gb
    echo [信息] 将使用32GB内存优化配置
)

:: 创建或更新.env文件
if exist .env.example (
    echo [信息] 使用.env.example模板创建.env文件...
    copy /y .env.example .env
) else if exist .env (
    echo [信息] 已找到.env文件，将进行更新...
) else (
    echo [错误] 找不到.env.example模板文件！
    goto :EOF
)

:: 更新.env文件配置
powershell -Command "(Get-Content .env) -replace 'MEMORY_CONFIG=default', 'MEMORY_CONFIG=%memory_config%' | Set-Content .env"

:: 解除相应配置的注释
set "pattern=# %memory_config%"
powershell -Command "$content = Get-Content .env -Raw; $section = $content -split '# [0-9]' | Where-Object { $_ -match '%pattern%' }; if($section) { $startIdx = $content.IndexOf($section); $endIdx = $content.IndexOf('# ', $startIdx + 1); if($endIdx -eq -1) { $endIdx = $content.Length }; $before = $content.Substring(0, $startIdx); $after = $content.Substring($endIdx); $modifiedSection = $section -replace '# ', ''; $newContent = $before + $modifiedSection + $after; $newContent | Set-Content .env; }"

echo [信息] 环境配置已更新为%memory_config%配置

:: 建立必要的目录
mkdir data\input data\output models 2>nul

:: 确保Hugging Face缓存目录存在
set "CACHE_DIR=%USERPROFILE%\.cache\huggingface"
mkdir "%CACHE_DIR%" 2>nul

echo [信息] 目录结构已创建

:: 检查示例数据集是否存在
if not exist "data\input\dataset.txt" (
    echo [警告] 未找到示例数据集文件 data\input\dataset.txt
    echo [信息] 提示: 您需要在训练前准备自己的数据集
) else (
    echo [信息] 已找到示例数据集文件
)

:: 提示用户下一步操作
echo [信息] 设置完成! 接下来的步骤:
echo [信息] 1. 将训练数据放入data\input\dataset.txt文件
echo [信息] 2. 构建Docker镜像: docker build -t chatglm-cpu-trainer .

if "%use_new_compose%"=="true" (
    echo [信息] 3. 开始训练: docker compose run train
    echo [信息] 4. 测试模型: docker compose run predict
) else (
    echo [信息] 3. 开始训练: docker-compose run train
    echo [信息] 4. 测试模型: docker-compose run predict
)

echo.
echo [信息] 自定义训练示例:
echo [信息] - 使用更少样本进行快速测试:
if "%use_new_compose%"=="true" (
    echo   MAX_SAMPLES=10 docker compose run train
) else (
    echo   MAX_SAMPLES=10 docker-compose run train
)

echo [信息] - 自定义提示词进行测试:
if "%use_new_compose%"=="true" (
    echo   PROMPT="请介绍一下深度学习技术" docker compose run predict
) else (
    echo   PROMPT="请介绍一下深度学习技术" docker-compose run predict
)

pause