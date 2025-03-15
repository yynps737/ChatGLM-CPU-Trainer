@echo off
setlocal EnableDelayedExpansion
title ChatGLM-CPU-Trainer 预测工具

:: 默认参数
set BASE_MODEL=THUDM/chatglm2-6b
set MODEL_PATH=/app/models/chatglm-lora
set PROMPT=请介绍一下人工智能的发展历史。
set QUANT=4bit
set MAX_LEN=2048
set OUTPUT_FILE=/app/data/output/prediction.txt

:: 参数解析
:arg_loop
if "%~1"=="" goto arg_done
if /i "%~1"=="-b" (
    set BASE_MODEL=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--base-model" (
    set BASE_MODEL=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="-m" (
    set MODEL_PATH=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--model-path" (
    set MODEL_PATH=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="-p" (
    set "PROMPT=%~2"
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--prompt" (
    set "PROMPT=%~2"
    shift & shift
    goto arg_loop
)
if /i "%~1"=="-q" (
    set QUANT=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--quantization" (
    set QUANT=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="-l" (
    set MAX_LEN=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--max-length" (
    set MAX_LEN=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="-o" (
    set OUTPUT_FILE=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--output" (
    set OUTPUT_FILE=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="-h" (
    goto show_help
)
if /i "%~1"=="--help" (
    goto show_help
)
echo 未知选项: %~1
goto show_help

:show_help
echo 使用方法: %0 [选项]
echo 选项:
echo   -b, --base-model MODEL   设置基础模型名称或路径 (默认: %BASE_MODEL%)
echo   -m, --model-path PATH    设置LoRA模型路径 (默认: %MODEL_PATH%)
echo   -p, --prompt PROMPT      设置提示文本 (默认: '%PROMPT%')
echo   -q, --quantization Q     设置量化类型 [4bit, 8bit, None] (默认: %QUANT%)
echo   -l, --max-length LEN     设置最大生成长度 (默认: %MAX_LEN%)
echo   -o, --output FILE        设置输出文件 (默认: %OUTPUT_FILE%)
echo   -h, --help               显示此帮助信息
goto :EOF

:arg_done

:: 执行预测命令
echo 开始预测...
echo 基础模型: %BASE_MODEL%
echo LoRA模型: %MODEL_PATH%
echo 提示: %PROMPT%
echo 输出文件: %OUTPUT_FILE%

:: 获取当前目录的绝对路径
for %%i in ("%~dp0..") do set PROJECT_DIR=%%~fi

:: 检查Docker是否运行
docker info >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] Docker引擎未运行，请先启动Docker！
    pause
    goto :EOF
)

:: 检查目录权限
echo [信息] 检查目录权限...
mkdir "%PROJECT_DIR%\data\output" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [警告] 无法创建输出目录，可能需要管理员权限
)

:: 检查模型是否存在
if not exist "%PROJECT_DIR%\models\chatglm-lora" (
    echo [警告] 未找到训练好的模型，将使用基础模型预测
)

:: 检查镜像是否存在
docker image inspect chatglm-cpu-trainer >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未找到chatglm-cpu-trainer镜像！请先构建镜像：
    echo docker build -t chatglm-cpu-trainer .
    pause
    goto :EOF
)

echo [信息] 开始模型加载和预测过程...
echo [信息] 如果是首次运行，将先下载模型文件(可能需要几分钟)
echo [信息] 请耐心等待，生成过程在Docker容器内进行

echo [信息] 启动Docker容器进行预测...
docker run --rm ^
    -v "%PROJECT_DIR%\data:/app/data" ^
    -v "%PROJECT_DIR%\models:/app/models" ^
    -v "%USERPROFILE%\.cache\huggingface:/root/.cache/huggingface" ^
    -e OMP_NUM_THREADS=4 ^
    -e MKL_NUM_THREADS=4 ^
    -e HF_ENDPOINT=https://hf-mirror.com ^
    chatglm-cpu-trainer ^
    python -m app.predict ^
    --base_model_path "%BASE_MODEL%" ^
    --model_path "%MODEL_PATH%" ^
    --prompt "%PROMPT%" ^
    --quantization "%QUANT%" ^
    --max_length "%MAX_LEN%" ^
    --output_file "%OUTPUT_FILE%"

if %ERRORLEVEL% NEQ 0 (
    echo [错误] 预测过程中出现错误！请检查日志获取详细信息。
) else (
    echo [成功] 预测完成！结果已保存到 %OUTPUT_FILE%

    :: 尝试显示结果
    echo [信息] 预测结果:
    echo ==================================
    type "%PROJECT_DIR%\data\output\prediction.txt" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [警告] 无法显示预测结果，请手动查看输出文件
    )
    echo ==================================
)

pause