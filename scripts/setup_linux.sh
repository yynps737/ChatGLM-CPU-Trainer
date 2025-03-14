#!/bin/bash
# ChatGLM训练环境配置脚本 - Linux版本

# 定义颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}    ChatGLM训练环境设置脚本 - Linux版本${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""

# 设置环境变量
echo -e "${YELLOW}设置环境变量...${NC}"
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export MKL_DYNAMIC=FALSE
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=""

# 检查Python版本
echo -e "${YELLOW}检查Python环境...${NC}"
PYTHON_VERSION=$(python --version 2>&1)
echo -e "${GREEN}检测到: ${PYTHON_VERSION}${NC}"

# 检查系统内存
echo -e "${YELLOW}检查系统内存...${NC}"
MEM_FREE_GB=$(free -g | grep Mem | awk '{print $7}')
echo -e "${GREEN}可用内存: ${MEM_FREE_GB} GB${NC}"

# 更新pip
echo -e "${YELLOW}更新pip...${NC}"
python -m pip install --upgrade pip

# 安装PyTorch
echo -e "${YELLOW}安装PyTorch...${NC}"
pip install torch torchvision

# 安装加速库
echo -e "${YELLOW}安装accelerate和bitsandbytes...${NC}"
pip install -U accelerate
pip install -U bitsandbytes>=0.39.0
if [ $? -ne 0 ]; then
    echo -e "${RED}bitsandbytes安装失败，将禁用量化功能${NC}"
fi

# 安装核心依赖
echo -e "${YELLOW}安装核心依赖...${NC}"
pip install transformers datasets peft evaluate scikit-learn pandas matplotlib

# 安装其他依赖（尝试安装deepspeed）
echo -e "${YELLOW}安装其他工具...${NC}"
pip install tensorboard tqdm psutil
pip install deepspeed

# 安装sentencepiece
echo -e "${YELLOW}安装sentencepiece...${NC}"
pip install sentencepiece

# 安装中文模型相关库
echo -e "${YELLOW}安装中文模型库...${NC}"
pip install modelscope
pip install icetk
pip install cpm_kernels

# 检查依赖安装状态
echo ""
echo -e "${GREEN}依赖安装状态:${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo -e "${RED}PyTorch: 未安装${NC}"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo -e "${RED}Transformers: 未安装${NC}"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" 2>/dev/null || echo -e "${RED}Accelerate: 未安装${NC}"
python -c "import bitsandbytes as bnb; print(f'BitsAndBytes: {bnb.__version__}')" 2>/dev/null || echo -e "${RED}BitsAndBytes: 未安装 - 量化功能不可用${NC}"
python -c "import peft; print(f'PEFT: {peft.__version__}')" 2>/dev/null || echo -e "${RED}PEFT: 未安装${NC}"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')" 2>/dev/null || echo -e "${RED}DeepSpeed: 未安装${NC}"

# 生成训练建议
echo ""
echo -e "${GREEN}环境设置完成!${NC}"
echo ""
echo -e "${YELLOW}基于系统内存 (${MEM_FREE_GB} GB) 推荐以下训练命令:${NC}"
echo ""

# 检查是否安装了bitsandbytes
python -c "import bitsandbytes" >/dev/null 2>&1
BNB_INSTALLED=$?

# 检查是否安装了deepspeed
python -c "import deepspeed" >/dev/null 2>&1
DS_INSTALLED=$?

# 基于内存和依赖生成建议命令
if [ $BNB_INSTALLED -ne 0 ]; then
    echo -e "${YELLOW}【量化不可用配置】${NC}"
    if [ $DS_INSTALLED -eq 0 ]; then
        echo "python ../train.py \\
  --model_name_or_path THUDM/chatglm3-6b \\
  --dataset_name uer/cluecorpussmall \\
  --auto_optimize_ds_config \\
  --use_lora \\
  --lora_r 4 \\
  --quantization None \\
  --max_seq_length 128 \\
  --max_samples 2000 \\
  --output_dir ../output/chatglm3-lora"
    else
        echo "python ../simple_train.py \\
  --model_name_or_path THUDM/chatglm3-6b \\
  --dataset_name uer/cluecorpussmall \\
  --use_lora \\
  --lora_r 4 \\
  --quantization None \\
  --max_seq_length 128 \\
  --max_samples 2000 \\
  --output_dir ../output/chatglm3-lora"
    fi
elif [ $MEM_FREE_GB -lt 8 ]; then
    echo -e "${YELLOW}【低内存配置】${NC}"
    if [ $DS_INSTALLED -eq 0 ]; then
        echo "python ../train.py \\
  --model_name_or_path THUDM/chatglm3-6b \\
  --dataset_name uer/cluecorpussmall \\
  --auto_optimize_ds_config \\
  --use_lora \\
  --lora_r 4 \\
  --quantization 4bit \\
  --max_seq_length 128 \\
  --max_samples 2000 \\
  --output_dir ../output/chatglm3-lora"
    else
        echo "python ../simple_train.py \\
  --model_name_or_path THUDM/chatglm3-6b \\
  --dataset_name uer/cluecorpussmall \\
  --use_lora \\
  --lora_r 4 \\
  --quantization 4bit \\
  --max_seq_length 128 \\
  --max_samples 2000 \\
  --output_dir ../output/chatglm3-lora"
    fi
else
    echo -e "${GREEN}【标准配置】${NC}"
    if [ $DS_INSTALLED -eq 0 ]; then
        echo "python ../train.py \\
  --model_name_or_path THUDM/chatglm3-6b \\
  --dataset_name uer/cluecorpussmall \\
  --auto_optimize_ds_config \\
  --use_lora \\
  --quantization 8bit \\
  --max_seq_length 256 \\
  --output_dir ../output/chatglm3-lora"
    else
        echo "python ../simple_train.py \\
  --model_name_or_path THUDM/chatglm3-6b \\
  --dataset_name uer/cluecorpussmall \\
  --use_lora \\
  --quantization 8bit \\
  --max_seq_length 256 \\
  --output_dir ../output/chatglm3-lora"
    fi
fi

echo ""