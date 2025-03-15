#!/bin/bash
# 训练模型的脚本

# 默认参数
MODEL="THUDM/chatglm2-6b"
DATASET="/app/data/input/dataset.txt"
OUTPUT="/app/models/chatglm-lora"
QUANT="4bit"
MAX_SEQ_LEN=64
MAX_SAMPLES=500
BATCH_SIZE=1
GRAD_ACCUM=16

# 打印使用方法
function print_usage {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -m, --model MODEL      设置模型名称或路径 (默认: $MODEL)"
    echo "  -d, --dataset DATASET  设置数据集路径 (默认: $DATASET)"
    echo "  -o, --output OUTPUT    设置输出目录 (默认: $OUTPUT)"
    echo "  -q, --quantization Q   设置量化类型 [4bit, 8bit, None] (默认: $QUANT)"
    echo "  -s, --seq-len LEN      设置最大序列长度 (默认: $MAX_SEQ_LEN)"
    echo "  --max-samples NUM      设置最大样本数 (默认: $MAX_SAMPLES)"
    echo "  -b, --batch-size SIZE  设置批量大小 (默认: $BATCH_SIZE)"
    echo "  -g, --grad-accum STEPS 设置梯度累积步数 (默认: $GRAD_ACCUM)"
    echo "  -h, --help             显示此帮助信息"
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -q|--quantization)
            QUANT="$2"
            shift 2
            ;;
        -s|--seq-len)
            MAX_SEQ_LEN="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -g|--grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "未知选项: $1"
            print_usage
            ;;
    esac
done

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT")"

# 执行训练命令
echo "开始训练..."
echo "模型: $MODEL"
echo "数据集: $DATASET"
echo "输出目录: $OUTPUT"
echo "量化级别: $QUANT"

docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    -e OMP_NUM_THREADS=4 \
    -e MKL_NUM_THREADS=4 \
    -e HF_ENDPOINT=https://hf-mirror.com \
    chatglm-cpu-trainer \
    python -m app.train \
    --model_name_or_path "$MODEL" \
    --dataset_path "$DATASET" \
    --output_dir "$OUTPUT" \
    --quantization "$QUANT" \
    --max_seq_length "$MAX_SEQ_LEN" \
    --max_samples "$MAX_SAMPLES" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM"

echo "训练完成！模型已保存到 $OUTPUT"