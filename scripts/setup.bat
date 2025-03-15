@echo off
setlocal EnableDelayedExpansion
title ChatGLM-CPU-Trainer 自动配置工具

echo [信息] 检查Docker环境...
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] Docker未安装! 请先安装Docker: https://docs.docker.com/get-docker/
    goto :EOF
)

:: 检查Windows长路径支持
echo [信息] 检查Windows长路径支持...
reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [警告] 无法检查长路径支持状态
) else (
    for /f "tokens=3" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" ^| findstr "LongPathsEnabled"') do (
        if "%%a"=="0x0" (
            echo [警告] Windows长路径支持未启用。项目路径较长时可能出现问题。
            echo [信息] 可以通过以管理员身份运行以下命令启用长路径支持:
            echo         reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d 1 /f
        ) else (
            echo [信息] Windows长路径支持已启用
        )
    )
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

:: 获取系统内存信息 - 使用更健壮的WMI方法
echo [信息] 检测系统内存...
set total_memory_gb=8
for /f "skip=1" %%a in ('wmic ComputerSystem get TotalPhysicalMemory') do (
    if not "%%a"=="" (
        set /a total_memory_gb=%%a/1024/1024/1024
        goto :memory_detected
    )
)
:: 备用方法：尝试使用systeminfo
if %total_memory_gb%==8 (
    for /f "tokens=4" %%a in ('systeminfo ^| findstr "Total Physical Memory"') do (
        set mem_str=%%a
        set mem_str=!mem_str:,=!
        set /a total_memory_mb=mem_str
        set /a total_memory_gb=total_memory_mb/1024
    )
)

:memory_detected
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

:: 根据选择的配置生成.env文件
echo [信息] 创建.env配置文件...
echo # 配置文件由setup.bat自动生成 > .env
echo MEMORY_CONFIG=%memory_config% >> .env

if "%memory_config%"=="4gb" (
    echo MEMORY_LIMIT=3.8G >> .env
    echo NUM_THREADS=2 >> .env
    echo QUANT_LEVEL=4bit >> .env
    echo MAX_SEQ_LEN=32 >> .env
    echo MAX_SAMPLES=30 >> .env
    echo LORA_R=4 >> .env
    echo BATCH_SIZE=1 >> .env
    echo GRAD_ACCUM=32 >> .env
    echo MAX_LENGTH=512 >> .env
    echo MONITOR_MEMORY=true >> .env
    echo MEMORY_CHECK_INTERVAL=30 >> .env
    echo PERFORMANCE_LOG_STEPS=50 >> .env
) else if "%memory_config%"=="8gb" (
    echo MEMORY_LIMIT=7.5G >> .env
    echo NUM_THREADS=4 >> .env
    echo QUANT_LEVEL=4bit >> .env
    echo MAX_SEQ_LEN=64 >> .env
    echo MAX_SAMPLES=200 >> .env
    echo LORA_R=8 >> .env
    echo BATCH_SIZE=1 >> .env
    echo GRAD_ACCUM=16 >> .env
    echo MAX_LENGTH=1024 >> .env
    echo MONITOR_MEMORY=true >> .env
    echo MEMORY_CHECK_INTERVAL=60 >> .env
    echo PERFORMANCE_LOG_STEPS=100 >> .env
) else if "%memory_config%"=="16gb" (
    echo MEMORY_LIMIT=15G >> .env
    echo NUM_THREADS=8 >> .env
    echo QUANT_LEVEL=8bit >> .env
    echo MAX_SEQ_LEN=128 >> .env
    echo MAX_SAMPLES=800 >> .env
    echo LORA_R=16 >> .env
    echo BATCH_SIZE=2 >> .env
    echo GRAD_ACCUM=8 >> .env
    echo MAX_LENGTH=2048 >> .env
    echo MONITOR_MEMORY=true >> .env
    echo MEMORY_CHECK_INTERVAL=120 >> .env
    echo PERFORMANCE_LOG_STEPS=200 >> .env
) else if "%memory_config%"=="32gb" (
    echo MEMORY_LIMIT=30G >> .env
    echo NUM_THREADS=16 >> .env
    echo QUANT_LEVEL=None >> .env
    echo MAX_SEQ_LEN=256 >> .env
    echo MAX_SAMPLES=2000 >> .env
    echo LORA_R=32 >> .env
    echo BATCH_SIZE=4 >> .env
    echo GRAD_ACCUM=4 >> .env
    echo MAX_LENGTH=4096 >> .env
    echo MONITOR_MEMORY=true >> .env
    echo MEMORY_CHECK_INTERVAL=180 >> .env
    echo PERFORMANCE_LOG_STEPS=300 >> .env
)

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