"""
ChatGLM 预测脚本 - 用于验证训练后的模型 (Docker优化版)
"""

import os
import argparse
import torch
from peft import PeftModel

# 导入工具函数
from app.utils import setup_logging, load_model_and_tokenizer

# 禁用CUDA，仅使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 设置日志
logger = setup_logging(log_file="/app/data/output/predict.log")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ChatGLM模型预测脚本 (Docker优化版)")

    parser.add_argument("--model_path", type=str, default="/app/models/chatglm-lora",
                        help="LoRA模型路径")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="基础模型路径")
    parser.add_argument("--prompt", type=str, default="请介绍一下人工智能的发展历史。",
                        help="测试提示")
    parser.add_argument("--quantization", type=str, default="None",
                        choices=["4bit", "8bit", "None"],
                        help="模型量化类型")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="生成的最大长度")
    parser.add_argument("--output_file", type=str, default="/app/data/output/prediction.txt",
                        help="输出文件路径")

    return parser.parse_args()


def load_model_for_inference(args):
    """加载基础模型和LoRA模型"""
    logger.info(f"加载基础模型: {args.base_model_path}")

    # 加载基础模型和分词器
    base_model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.quantization)

    # 加载LoRA模型
    logger.info(f"加载LoRA模型: {args.model_path}")
    model = PeftModel.from_pretrained(
        base_model,
        args.model_path
    )
    logger.info("LoRA模型加载成功!")

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length):
    """生成回复"""
    logger.info(f"提示: {prompt}")

    try:
        # 使用chat方法生成回复
        response, history = model.chat(
            tokenizer,
            prompt,
            history=[],
            max_length=max_length
        )
        return response
    except Exception as e:
        logger.error(f"生成回复时出错: {e}")

        # 尝试备用生成方法
        logger.warning("尝试使用备用生成方法...")
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e2:
            logger.error(f"备用生成方法也失败: {e2}")
            raise


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置CPU环境变量
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "4")

    logger.info("=" * 50)
    logger.info("ChatGLM 模型预测脚本 (Docker优化版)")
    logger.info("=" * 50)
    logger.info(f"基础模型: {args.base_model_path}")
    logger.info(f"LoRA模型: {args.model_path}")
    logger.info(f"量化级别: {args.quantization}")
    logger.info("=" * 50)

    # 加载模型和分词器
    model, tokenizer = load_model_for_inference(args)

    # 设置模型为评估模式
    model.eval()

    # 生成回复
    logger.info("开始生成回复...")
    response = generate_response(model, tokenizer, args.prompt, args.max_length)

    # 输出结果
    logger.info("生成完成!")
    print("\n" + "=" * 50)
    print("生成的回复:")
    print("-" * 50)
    print(response)
    print("=" * 50)

    # 保存到文件
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(f"提示: {args.prompt}\n\n")
        f.write(f"回复:\n{response}")

    logger.info(f"结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()