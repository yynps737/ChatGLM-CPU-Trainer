import os
import argparse
import json
import torch
import logging
from transformers import set_seed
from src.models.model_utils import load_base_model, load_peft_model

# 禁用CUDA，仅使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估语言模型")

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--base_model_path", type=str, default=None, help="基础模型路径（对PEFT模型使用）")
    parser.add_argument("--is_peft_model", action="store_true", help="是否为PEFT模型")

    # 评估参数
    parser.add_argument("--input_file", type=str, default=None, help="输入提示文件路径")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="输出结果文件路径")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成的最大新token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样参数")
    parser.add_argument("--top_k", type=int, default=50, help="top-k采样参数")

    # 量化参数
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "8bit", "4bit"],
                        help="模型量化级别 (8bit, 4bit, 或不量化)")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


def read_prompts(input_file):
    """读取提示文件"""
    prompts = []
    if input_file and os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    prompts.append(line.strip())
    else:
        # 使用默认提示
        prompts = [
            "请解释深度学习的基本原理。",
            "如何使用PyTorch实现一个简单的神经网络？",
            "写一个关于月亮的短诗。"
        ]
        logger.info(f"使用默认提示：{prompts}")

    return prompts


def generate_responses(model, tokenizer, prompts, args):
    """生成响应"""
    results = []

    # 将模型设置为评估模式
    model.eval()

    for i, prompt in enumerate(prompts):
        logger.info(f"处理提示 {i + 1}/{len(prompts)}")

        # 准备输入
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.device)

        # 配置生成参数
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }

        # 生成响应
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                **gen_kwargs
            )

        # 解码响应
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 如果模型没有移除原始提示，尝试分割响应
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()

        results.append({
            "prompt": prompt,
            "generated": response
        })

        # 打印结果示例
        logger.info(f"提示: {prompt[:50]}...")
        logger.info(f"响应: {response[:100]}...")

    return results


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载模型
    logger.info("加载模型...")
    if args.is_peft_model:
        if not args.base_model_path:
            raise ValueError("使用PEFT模型时必须提供base_model_path")
        model, tokenizer = load_peft_model(
            args.base_model_path,
            args.model_path,
            quantization=args.quantization
        )
    else:
        model, tokenizer = load_base_model(
            args.model_path,
            quantization=args.quantization
        )

    # 读取提示
    logger.info("读取提示...")
    prompts = read_prompts(args.input_file)

    # 生成响应
    logger.info("生成响应...")
    results = generate_responses(model, tokenizer, prompts, args)

    # 保存结果
    logger.info(f"保存结果到 {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    logger.info("评估完成！")


if __name__ == "__main__":
    main()