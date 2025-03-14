@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ===============================================
echo    ChatGLM CPU Training Script - Simple Version
echo ===============================================
echo.

rem Set CPU environment variables
echo Setting environment variables...
for /f "tokens=*" %%a in ('powershell -command "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty NumberOfLogicalProcessors"') do set CPU_CORES=%%a
echo Detected %CPU_CORES% logical processor cores

set OMP_NUM_THREADS=%CPU_CORES%
set MKL_NUM_THREADS=%CPU_CORES%
set HF_ENDPOINT=https://hf-mirror.com
set CUDA_VISIBLE_DEVICES=

rem Check memory
echo Checking system memory...
for /f "tokens=*" %%a in ('powershell -command "[Math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)"') do set MEM_FREE_GB=%%a
echo Available memory: %MEM_FREE_GB% GB

rem Create output directory
echo Creating output directory...
if not exist "..\output\chatglm-lora" mkdir "..\output\chatglm-lora"

rem Set ultra low resource configuration
set MODEL=THUDM/chatglm2-6b
set DATASET=uer/cluecorpussmall
set MAX_SEQ=64
set MAX_SAMPLES=500
set LORA_R=4

echo.
echo Training parameters (ultra low resource config):
echo - Model: %MODEL%
echo - Dataset: %DATASET%
echo - Max sequence length: %MAX_SEQ%
echo - Max samples: %MAX_SAMPLES%
echo - LoRA rank: %LORA_R%
echo.

rem Start training
echo Starting training, log will be saved to ..\output\chatglm-lora\train_log.txt
echo Using ultra low resource configuration for limited memory...

python ..\simple_train.py ^
  --model_name_or_path %MODEL% ^
  --dataset_name %DATASET% ^
  --use_lora ^
  --lora_r %LORA_R% ^
  --quantization 4bit ^
  --max_seq_length %MAX_SEQ% ^
  --max_samples %MAX_SAMPLES% ^
  --per_device_train_batch_size 1 ^
  --gradient_accumulation_steps 16 ^
  --output_dir ..\output\chatglm-lora > ..\output\chatglm-lora\train_log.txt 2>&1

rem Check if training was successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Training completed successfully!
    echo Model saved to ..\output\chatglm-lora
    echo.
    echo Test model:
    echo python ..\test_model.py --model_path ..\output\chatglm-lora --base_model_path %MODEL% --is_peft_model --quantization 4bit --prompt "Please explain deep learning briefly"
) else (
    echo.
    echo Training failed! Please check log file: ..\output\chatglm-lora\train_log.txt
    echo.
    echo Common errors:
    echo 1. Out of memory error (OOM) - Try reducing sample count and sequence length
    echo 2. BitsAndBytes error - Run the following command to install alternative version:
    echo    pip uninstall bitsandbytes-windows
    echo    pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
    echo.
    echo Tail of log file:
    echo ----------------
    powershell -command "Get-Content ..\output\chatglm-lora\train_log.txt -Tail 20"
)

echo.
echo Press any key to exit...
pause > nul