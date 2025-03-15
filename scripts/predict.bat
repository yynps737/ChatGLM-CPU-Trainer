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

echo 预测完成！结果已保存到 %OUTPUT_FILE%
pause