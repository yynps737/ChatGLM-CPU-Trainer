@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ===============================================
echo    ChatGLM CPU训练脚本 - Windows版本 (Python 3.10)
echo ===============================================
echo.

REM 设置CPU相关环境变量 - 动态获取CPU核心数
echo 设置环境变量...
for /f "tokens=*" %%a in ('powershell -command "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty NumberOfLogicalProcessors"') do set CPU_CORES=%%a
echo 检测到 %CPU_CORES% 个逻辑处理器核心

set OMP_NUM_THREADS=%CPU_CORES%
set MKL_NUM_THREADS=%CPU_CORES%
set MKL_DYNAMIC=FALSE
set OMP_SCHEDULE=STATIC
set OMP_PROC_BIND=CLOSE
set HF_ENDPOINT=https://hf-mirror.com
set CUDA_VISIBLE_DEVICES=

REM 检查Python
echo 检查Python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo 错误: Python未安装或未添加到PATH
    goto :end
)

REM 检查内存
echo 检查系统内存...
for /f "tokens=*" %%a in ('powershell -command "[Math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)"') do set MEM_FREE_GB=%%a
echo 可用内存: %MEM_FREE_GB% GB

REM 检测是否使用Python 3.8-3.13
python -c "import sys; sys.exit(0 if (sys.version_info.major == 3 and sys.version_info.minor >= 8 and sys.version_info.minor <= 13) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 警告: 当前Python版本可能不兼容，推荐使用Python 3.8-3.13
)

REM 检查bitsandbytes是否安装
python -c "import bitsandbytes" >nul 2>&1
set BNB_INSTALLED=%ERRORLEVEL%

REM 设置训练参数
set MAX_SEQ=256
set QUANT_ARG=--quantization 8bit
set LORA_R=8
set MAX_SAMPLES=

if %BNB_INSTALLED% NEQ 0 (
    echo 量化库未安装或不可用，将使用非量化模式
    set QUANT_ARG=--quantization None
) else (
    REM 测试bitsandbytes量化功能是否可用
    python -c "import torch; from bitsandbytes.nn import Linear8bitLt; x = torch.randn(1, 10); layer = Linear8bitLt(10, 10, has_fp16_weights=False); y = layer(x); print('OK')" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo bitsandbytes量化测试失败，将使用非量化模式
        set QUANT_ARG=--quantization None
    )
)

if %MEM_FREE_GB% LSS 8 (
    echo 检测到低内存环境 (小于8GB)，使用保守设置
    set MAX_SEQ=128
    if not "%QUANT_ARG%"=="--quantization None" set QUANT_ARG=--quantization 4bit
    set LORA_R=4
    set MAX_SAMPLES=--max_samples 2000
) else if %MEM_FREE_GB% LSS 12 (
    echo 检测到中等内存环境 (8-12GB)，调整参数
    if not "%QUANT_ARG%"=="--quantization None" set QUANT_ARG=--quantization 4bit
    set MAX_SAMPLES=--max_samples 5000
)

REM 创建输出目录
echo 创建输出目录...
if not exist "..\output\chatglm3-lora" mkdir "..\output\chatglm3-lora"

REM 检测是否存在DeepSpeed以选择合适的训练脚本
python -c "import deepspeed" >nul 2>&1
set DS_INSTALLED=%ERRORLEVEL%

if %DS_INSTALLED% EQU 0 (
    echo 检测到DeepSpeed，使用标准训练脚本...
    set TRAIN_SCRIPT=..\train.py
    set DS_ARG=--auto_optimize_ds_config
) else (
    echo 未检测到DeepSpeed，使用简化训练脚本...
    set TRAIN_SCRIPT=..\simple_train.py
    set DS_ARG=
)

REM 显示训练参数
echo.
echo 训练参数:
echo - 脚本: %TRAIN_SCRIPT%
echo - 最大序列长度: %MAX_SEQ%
echo - 量化设置: %QUANT_ARG%
echo - LoRA rank: %LORA_R%
echo - 样本限制: %MAX_SAMPLES%
echo.

REM 开始训练
echo 开始训练，日志将保存到 ..\output\chatglm3-lora\train_log.txt
echo 正在启动...

python %TRAIN_SCRIPT% ^
  --model_name_or_path THUDM/chatglm3-6b ^
  --dataset_name uer/cluecorpussmall ^
  %DS_ARG% ^
  --use_lora ^
  --lora_r %LORA_R% ^
  %QUANT_ARG% ^
  --max_seq_length %MAX_SEQ% ^
  %MAX_SAMPLES% ^
  --per_device_train_batch_size 1 ^
  --gradient_accumulation_steps 32 ^
  --output_dir ..\output\chatglm3-lora > ..\output\chatglm3-lora\train_log.txt 2>&1

REM 检查训练是否成功
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo 训练失败！请查看日志文件 ..\output\chatglm3-lora\train_log.txt
    echo 常见问题:
    echo 1. 内存不足 - 请尝试减少参数: --max_samples 1000 --max_seq_length 64
    echo 2. 依赖问题 - 请运行 ..\scripts\setup_windows.bat 安装所有依赖
    echo 3. 详细错误信息:
    powershell -command "Get-Content ..\output\chatglm3-lora\train_log.txt | Select-String -Pattern 'Error|Exception|错误'"
) else (
    color 0A
    echo 训练完成！输出保存在 ..\output\chatglm3-lora
)

:end
echo.
pause