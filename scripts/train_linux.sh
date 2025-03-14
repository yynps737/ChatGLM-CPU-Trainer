#!/bin/bash
# ChatGLM CPU训练脚本 - Linux版本

# 定义颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}    ChatGLM CPU训练脚本 - Linux版本${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""

# 设置CPU相关环境变量
echo -e "${YELLOW}设置环境变量...${NC}"
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export MKL_DYNAMIC=FALSE
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=""

# 清理系统缓存（如果有root权限）
echo -e "${YELLOW}尝试清理系统缓存...${NC}"
if [ "$(id -u)" -eq 0 ]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches
    echo -e "${GREEN}系统缓存已清理${NC}"
else
    echo -e "${YELLOW}无root权限，跳过缓存清理${NC}"
fi

# 检查内存状态
echo -e "${YELLOW}检查系统内存...${NC}"
MEM_FREE_GB=$(free -g | grep Mem | awk '{print $7}')
echo -e "${GREEN}可用内存: ${MEM_FREE_GB} GB${NC}"

# 检查bitsandbytes是否安装
python -c "import bitsandbytes" >/dev/null 2>&1
BNB_INSTALLED=$?

# 设置训练参数
MAX_SEQ=256
QUANT_ARG="--quantization 8bit"
LORA_R=8
MAX_SAMPLES=""

if [ $BNB_INSTALLED -ne 0 ]; then
    echo -e "${YELLOW}量化库未安装，将使用非量化模式${NC}"
    QUANT_ARG="--quantization None"
elif [ $MEM_FREE_GB -lt 8 ]; then
    echo -e "${YELLOW}检测到低内存环境 (小于8GB)，使用保守设置${NC}"
    MAX_SEQ=128
    QUANT_ARG="--quantization 4bit"
    LORA_R=4
    MAX_SAMPLES="--max_samples 2000"
elif [ $MEM_FREE_GB -lt 12 ]; then
    echo -e "${YELLOW}检测到中等内存环境 (8-12GB)，调整参数${NC}"
    QUANT_ARG="--quantization 4bit"
    MAX_SAMPLES="--max_samples 5000"
fi

# 创建输出目录
echo -e "${YELLOW}创建输出目录...${NC}"
mkdir -p ../output/chatglm3-lora

# 检测是否存在DeepSpeed以选择合适的训练脚本
python -c "import deepspeed" >/dev/null 2>&1
DS_INSTALLED=$?

if [ $DS_INSTALLED -eq 0 ]; then
    echo -e "${GREEN}检测到DeepSpeed，使用标准训练脚本...${NC}"
    TRAIN_SCRIPT="../train.py"
else
    echo -e "${YELLOW}未检测到DeepSpeed，使用简化训练脚本...${NC}"
    TRAIN_SCRIPT="../simple_train.py"
fi

# 显示训练参数
echo ""
echo -e "${GREEN}训练参数:${NC}"
echo "- 脚本: $TRAIN_SCRIPT"
echo "- 最大序列长度: $MAX_SEQ"
echo "- 量化设置: $QUANT_ARG"
echo "- LoRA rank: $LORA_R"
echo "- 样本限制: $MAX_SAMPLES"
echo ""

# 开始训练
echo -e "${GREEN}开始训练，日志将保存到 ../output/chatglm3-lora/train_log.txt${NC}"
echo -e "${YELLOW}正在启动...${NC}"

time python $TRAIN_SCRIPT \
  --model_name_or_path THUDM/chatglm3-6b \
  --dataset_name uer/cluecorpussmall \
  --use_lora \
  --lora_r $LORA_R \
  $QUANT_ARG \
  --max_seq_length $MAX_SEQ \
  $MAX_SAMPLES \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --output_dir ../output/chatglm3-lora 2>&1 | tee ../output/chatglm3-lora/train_log.txt

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo -e "${RED}训练失败！请查看日志文件 ../output/chatglm3-lora/train_log.txt${NC}"
    echo -e "${YELLOW}常见问题:${NC}"
    echo "1. 内存不足 - 请尝试减少参数: --max_samples 1000 --max_seq_length 64"
    echo "2. 依赖问题 - 请运行 ../scripts/setup_linux.sh 安装所有依赖"
    echo "3. 详细错误信息:"
    grep -E "Error|Exception|错误" ../output/chatglm3-lora/train_log.txt
else
    echo -e "${GREEN}训练完成！输出保存在 ../output/chatglm3-lora${NC}"
fi

echo ""