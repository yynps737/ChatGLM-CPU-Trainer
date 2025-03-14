import os
import argparse
import torch
import logging
from src.models.model_utils import load_base_model, load_peft_model, is_chatglm_model
from transformers import set_seed

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
    parser = argparse.ArgumentParser(description="测试模型加载和生成")
    parser.add_argument("--model_path", type=str, default="THUDM/chatglm3-6b", help="模型路径")
    parser.add_argument("--is_peft_model", action="store_true", help="是否为PEFT模型")
    parser.add_argument("--base_model_path", type=str, default=None, help="基础模型路径（对PEFT模型使用）")
    parser.add_argument("--quantization", type=str, default="8bit", choices=[None, "8bit", "4bit"], help="量化级别")
    parser.add_argument("--prompt", type=str, default="请给我解释一下深度学习是什么？", help="测试提示")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="生成的最大token数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 记录CPU环境
    logger.info("运行在CPU模式下，已禁用CUDA")
    logger.info(f"CPU线程设置: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '未设置')}")
    logger.info(f"MKL线程设置: MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '未设置')}")

    # 加载模型
    logger.info(f"正在加载模型: {args.model_path}")
    if args.is_peft_model:
        if not args.base_model_path:
            raise ValueError("使用PEFT模型时必须提供base_model_path")
        model, tokenizer = load_peft_model(args.base_model_path, args.model_path, args.quantization)
    else:
        model, tokenizer = load_base_model(args.model_path, quantization=args.quantization)

    # 打印模型信息
    logger.info(f"模型类型: {type(model).__name__}")
    logger.info(f"模型设备: {model.device}")

    # 准备输入
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

    # 生成文本
    logger.info(f"使用提示: {args.prompt}")
    logger.info("正在生成回复...")

    with torch.no_grad():
        is_glm = is_chatglm_model(args.model_path)

        if is_glm:
            # ChatGLM特定的生成方法
            logger.info("使用ChatGLM特定的生成方法")
            response, history = model.chat(
                tokenizer,
                args.prompt,
                history=[],
                max_length=args.max_new_tokens + len(inputs["input_ids"][0]),
                temperature=0.7,
                top_p=0.9
            )
            reply = response
        else:
            # 标准生成
            logger.info("使用标准生成方法")
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取回复（移除原始提示）
            if generated_text.startswith(args.prompt):
                reply = generated_text[len(args.prompt):].strip()
            else:
                reply = generated_text.strip()

    print("\n生成的回复:")
    print("-" * 50)
    print(reply)
    print("-" * 50)
    print(f"生成了 {len(reply)} 个字符")

if __name__ == "__main__":
    main()