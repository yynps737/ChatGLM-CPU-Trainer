#!/bin/bash
# ChatGLM 问答能力微调示例 - Linux版本

# 定义颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}    ChatGLM 问答能力微调示例 - Linux版本${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""

# 设置环境变量
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=""

# 检查bitsandbytes是否安装
python -c "import bitsandbytes" >/dev/null 2>&1
BNB_INSTALLED=$?

# 设置量化参数
if [ $BNB_INSTALLED -ne 0 ]; then
    QUANT_ARG="--quantization None"
    echo -e "${YELLOW}警告: bitsandbytes未安装，将不使用量化${NC}"
else
    QUANT_ARG="--quantization 8bit"
fi

# 检测DeepSpeed
python -c "import deepspeed" >/dev/null 2>&1
if [ $? -eq 0 ]; then
    TRAIN_SCRIPT="../train.py"
    DS_ARG="--auto_optimize_ds_config"
else
    TRAIN_SCRIPT="../simple_train.py"
    DS_ARG=""
fi

# 创建输出目录
mkdir -p ../output/chatglm3-qa

echo -e "${GREEN}开始问答能力微调...${NC}"
echo -e "${YELLOW}使用脚本: ${TRAIN_SCRIPT}${NC}"
echo ""

python $TRAIN_SCRIPT \
  --model_name_or_path THUDM/chatglm3-6b \
  --dataset_name THUDM/webglm \
  --instruction_format \
  --instruction_column question \
  --output_column answer \
  --max_seq_length 384 \
  $DS_ARG \
  --use_lora \
  --lora_r 8 \
  $QUANT_ARG \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 5e-5 \
  --output_dir ../output/chatglm3-qa

echo ""
if [ $? -eq 0 ]; then
    echo -e "${GREEN}微调完成！模型已保存到 ../output/chatglm3-qa${NC}"
    echo ""
    echo -e "${YELLOW}测试模型示例:${NC}"
    echo "python ../test_model.py --model_path ../output/chatglm3-qa --base_model_path THUDM/chatglm3-6b --is_peft_model $QUANT_ARG --prompt \"中国有多少个省份？\""
else
    echo -e "${RED}微调失败，请查看上面的错误信息${NC}"
fi

echo ""