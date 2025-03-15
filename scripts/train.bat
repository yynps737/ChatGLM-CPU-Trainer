@echo off
setlocal EnableDelayedExpansion
title ChatGLM-CPU-Trainer 训练工具

:: 默认参数
set MODEL=THUDM/chatglm2-6b
set DATASET=/app/data/input/dataset.txt
set OUTPUT=/app/models/chatglm-lora
set QUANT=4bit
set MAX_SEQ_LEN=64
set MAX_SAMPLES=500
set BATCH_SIZE=1
set GRAD_ACCUM=16

:: 参数解析
:arg_loop
if "%~1"=="" goto arg_done
if /i "%~1"=="-m" (
    set MODEL=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--model" (
    set MODEL=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="-d" (
    set DATASET=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--dataset" (
    set DATASET=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="-o" (
    set OUTPUT=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--output" (
    set OUTPUT=%~2
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
if /i "%~1"=="-s" (
    set MAX_SEQ_LEN=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--seq-len" (
    set MAX_SEQ_LEN=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--max-samples" (
    set MAX_SAMPLES=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="-b" (
    set BATCH_SIZE=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--batch-size" (
    set BATCH_SIZE=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="-g" (
    set GRAD_ACCUM=%~2
    shift & shift
    goto arg_loop
)
if /i "%~1"=="--grad-accum" (
    set GRAD_ACCUM=%~2
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
echo   -m, --model MODEL      设置模型名称或路径 (默认: %MODEL%)
echo   -d, --dataset DATASET  设置数据集路径 (默认: %DATASET%)
echo   -o, --output OUTPUT    设置输出目录 (默认: %OUTPUT%)
echo   -q, --quantization Q   设置量化类型 [4bit, 8bit, None] (默认: %QUANT%)
echo   -s, --seq-len LEN      设置最大序列长度 (默认: %MAX_SEQ_LEN%)
echo   --max-samples NUM      设置最大样本数 (默认: %MAX_SAMPLES%)
echo   -b, --batch-size SIZE  设置批量大小 (默认: %BATCH_SIZE%)
echo   -g, --grad-accum STEPS 设置梯度累积步数 (默认: %GRAD_ACCUM%)
echo   -h, --help             显示此帮助信息
goto :EOF

:arg_done

:: 执行训练命令
echo 开始训练...
echo 模型: %MODEL%
echo 数据集: %DATASET%
echo 输出目录: %OUTPUT%
echo 量化级别: %QUANT%

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
    python -m app.train ^
    --model_name_or_path "%MODEL%" ^
    --dataset_path "%DATASET%" ^
    --output_dir "%OUTPUT%" ^
    --quantization "%QUANT%" ^
    --max_seq_length "%MAX_SEQ_LEN%" ^
    --max_samples "%MAX_SAMPLES%" ^
    --per_device_train_batch_size "%BATCH_SIZE%" ^
    --gradient_accumulation_steps "%GRAD_ACCUM%"

echo 训练完成！模型已保存到 %OUTPUT%
pause