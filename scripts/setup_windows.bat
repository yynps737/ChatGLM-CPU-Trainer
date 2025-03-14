@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ===============================================
echo    ChatGLM训练环境设置脚本 - Windows版本
echo ===============================================
echo.

REM 使用Windows原生颜色
color 0A

REM 获取CPU核心数
for /f "tokens=*" %%a in ('powershell -command "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty NumberOfLogicalProcessors"') do set CPU_CORES=%%a
echo 检测到 %CPU_CORES% 个逻辑处理器核心

REM 检查Python版本
echo 检查Python环境...
for /f "tokens=*" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo 检测到: %PYTHON_VERSION%

REM 验证Python版本
python -c "import sys; sys.exit(0 if (sys.version_info.major == 3 and sys.version_info.minor >= 8 and sys.version_info.minor <= 13) else 1)"
if %ERRORLEVEL% NEQ 0 (
    color 0E
    echo 警告: 推荐使用Python 3.8-3.13版本，当前版本可能存在兼容性问题
    pause
)

REM 检查内存状态 - 使用PowerShell代替wmic
echo 检查系统内存...
for /f "tokens=*" %%a in ('powershell -command "[Math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)"') do set MEM_FREE_GB=%%a
echo 可用内存: %MEM_FREE_GB% GB

REM 更新pip和基础工具
echo 更新pip和基础工具...
python -m pip install --upgrade pip setuptools wheel

REM 安装PyTorch
echo 安装PyTorch CPU版本...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

REM 安装accelerate
echo 安装accelerate...
pip install -U accelerate

REM 安装bitsandbytes - Windows版本
echo 安装bitsandbytes Windows版本...
pip install bitsandbytes-windows
if %ERRORLEVEL% NEQ 0 (
    color 0E
    echo 警告: bitsandbytes-windows安装失败，将尝试备用方法
    pip install bitsandbytes-windows-cpu
    if %ERRORLEVEL% NEQ 0 (
        echo 两种方法均失败，量化功能将不可用
    )
)

REM 安装核心依赖
echo 安装核心依赖...
pip install transformers datasets peft evaluate scikit-learn pandas matplotlib

REM 安装其他工具
echo 安装其他工具...
pip install tensorboard tqdm psutil

REM 尝试安装sentencepiece (预编译wheel优先)
echo 尝试安装sentencepiece...
pip install sentencepiece --only-binary :all:
if %ERRORLEVEL% NEQ 0 (
    echo 尝试直接安装sentencepiece...
    pip install sentencepiece
    if %ERRORLEVEL% NEQ 0 (
        echo 警告: sentencepiece安装失败，某些功能可能不可用
    )
)

REM 安装中文模型相关库
echo 安装中文模型库...
pip install modelscope
pip install icetk

REM 检查已安装依赖
echo.
echo 依赖安装状态:
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 echo PyTorch: 未安装
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 echo Transformers: 未安装
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 echo Accelerate: 未安装

REM 检查bitsandbytes
python -c "import bitsandbytes as bnb; print(f'BitsAndBytes: {bnb.__version__}')" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo BitsAndBytes已安装，测试量化功能...
    python -c "import torch; from bitsandbytes.nn import Linear8bitLt; x = torch.randn(1, 10); layer = Linear8bitLt(10, 10, has_fp16_weights=False); print('量化层测试成功')" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo 警告: BitsAndBytes已安装但量化功能测试失败
    )
) else (
    echo BitsAndBytes: 未安装 - 量化功能不可用
)

python -c "import peft; print(f'PEFT: {peft.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 echo PEFT: 未安装

REM 生成训练建议
echo.
echo 环境设置完成!
echo.
echo 基于系统内存 (%MEM_FREE_GB% GB) 和CPU核心数 (%CPU_CORES%) 推荐以下训练命令:
echo.

REM 检查是否安装了bitsandbytes
python -c "import bitsandbytes" >nul 2>&1
set BNB_INSTALLED=%ERRORLEVEL%

if %BNB_INSTALLED% NEQ 0 (
    echo 【量化不可用配置】
    echo python ..\simple_train.py ^
  --model_name_or_path THUDM/chatglm3-6b ^
  --dataset_name uer/cluecorpussmall ^
  --use_lora ^
  --lora_r 4 ^
  --quantization None ^
  --max_seq_length 128 ^
  --max_samples 2000 ^
  --output_dir ..\output\chatglm3-lora
) else if %MEM_FREE_GB% LSS 8 (
    echo 【低内存配置】
    echo python ..\simple_train.py ^
  --model_name_or_path THUDM/chatglm3-6b ^
  --dataset_name uer/cluecorpussmall ^
  --use_lora ^
  --lora_r 4 ^
  --quantization 4bit ^
  --max_seq_length 128 ^
  --max_samples 2000 ^
  --output_dir ..\output\chatglm3-lora
) else (
    echo 【标准配置】
    echo python ..\simple_train.py ^
  --model_name_or_path THUDM/chatglm3-6b ^
  --dataset_name uer/cluecorpussmall ^
  --use_lora ^
  --quantization 8bit ^
  --max_seq_length 256 ^
  --output_dir ..\output\chatglm3-lora
)

echo.
echo 设置的环境变量:
echo set OMP_NUM_THREADS=%CPU_CORES%
echo set MKL_NUM_THREADS=%CPU_CORES%
echo.

echo 按任意键退出...
pause > nul