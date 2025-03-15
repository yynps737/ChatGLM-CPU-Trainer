#!/bin/bash
# 模型预测脚本

# 默认参数
BASE_MODEL="THUDM/chatglm2-6b"
MODEL_PATH="/app/models/chatglm-lora"
PROMPT="请介绍一下人工智能的发展历史。"
QUANT="4bit"
MAX_LEN=2048
OUTPUT_FILE="/app/data/output/prediction.txt"

# 打印使用方法
function print_usage {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -b, --base-model MODEL   设置基础模型名称或路径 (默认: $BASE_MODEL)"
    echo "  -m, --model-path PATH    设置LoRA模型路径 (默认: $MODEL_PATH)"
    echo "  -p, --prompt PROMPT      设置提示文本 (默认: '$PROMPT')"
    echo "  -q, --quantization Q     设置量化类型 [4bit, 8bit, None] (默认: $QUANT)"
    echo "  -l, --max-length LEN     设置最大生成长度 (默认: $MAX_LEN)"
    echo "  -o, --output FILE        设置输出文件 (默认: $OUTPUT_FILE)"
    echo "  -h, --help               显示此帮助信息"
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -q|--quantization)
            QUANT="$2"
            shift 2
            ;;
        -l|--max-length)
            MAX_LEN="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
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
mkdir -p "$(dirname "$OUTPUT_FILE")"

# 执行预测命令
echo "开始预测..."
echo "基础模型: $BASE_MODEL"
echo "LoRA模型: $MODEL_PATH"
echo "提示: $PROMPT"
echo "输出文件: $OUTPUT_FILE"

docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    -e OMP_NUM_THREADS=4 \
    -e MKL_NUM_THREADS=4 \
    -e HF_ENDPOINT=https://hf-mirror.com \
    chatglm-cpu-trainer \
    python -m app.predict \
    --base_model_path "$BASE_MODEL" \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --quantization "$QUANT" \
    --max_length "$MAX_LEN" \
    --output_file "$OUTPUT_FILE"

echo "预测完成！结果已保存到 $OUTPUT_FILE"