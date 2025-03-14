@echo off
echo ===============================================
echo    ChatGLM 指令微调示例 - Windows版本
echo ===============================================
echo.

:: 设置环境变量
set OMP_NUM_THREADS=32
set MKL_NUM_THREADS=32
set HF_ENDPOINT=https://hf-mirror.com
set CUDA_VISIBLE_DEVICES=

:: 检查bitsandbytes是否安装
python -c "import bitsandbytes" >nul 2>&1
set BNB_INSTALLED=%ERRORLEVEL%

:: 设置量化参数
if %BNB_INSTALLED% NEQ 0 (
    set QUANT_ARG=--quantization None
    echo 警告: bitsandbytes未安装，将不使用量化
) else (
    set QUANT_ARG=--quantization 8bit
)

:: 检测DeepSpeed
python -c "import deepspeed" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set TRAIN_SCRIPT=..\train.py
    set DS_ARG=--auto_optimize_ds_config
) else (
    set TRAIN_SCRIPT=..\simple_train.py
    set DS_ARG=
)

:: 创建输出目录
if not exist "..\output\chatglm2-alpaca" mkdir "..\output\chatglm2-alpaca"

echo 开始指令微调...
echo 使用脚本: %TRAIN_SCRIPT%
echo.

python %TRAIN_SCRIPT% ^
  --model_name_or_path THUDM/chatglm2-6b ^
  --dataset_name yahma/alpaca-cleaned ^
  --instruction_format ^
  --instruction_column instruction ^
  --input_column input ^
  --output_column output ^
  --max_seq_length 512 ^
  %DS_ARG% ^
  --use_lora ^
  %QUANT_ARG% ^
  --num_train_epochs 3 ^
  --per_device_train_batch_size 1 ^
  --gradient_accumulation_steps 32 ^
  --learning_rate 5e-5 ^
  --output_dir ..\output\chatglm2-alpaca

echo.
if %ERRORLEVEL% EQU 0 (
    echo 微调完成！模型已保存到 ..\output\chatglm2-alpaca
    echo.
    echo 测试模型示例:
    echo python ..\test_model.py --model_path ..\output\chatglm2-alpaca --base_model_path THUDM/chatglm2-6b --is_peft_model %QUANT_ARG% --prompt "告诉我宇宙的奥秘"
) else (
    echo 微调失败，请查看上面的错误信息
)

pause