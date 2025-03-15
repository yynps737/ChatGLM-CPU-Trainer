"""
ChatGLM 简化版模型测试脚本 - 用于验证训练后的模型
"""

import os
import argparse
import torch
import logging
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

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
    parser = argparse.ArgumentParser(description="ChatGLM模型测试脚本")

    parser.add_argument("--model_path", type=str, required=True,
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

    return parser.parse_args()

def load_model(args):
    """加载基础模型和LoRA模型"""
    logger.info(f"加载基础模型: {args.base_model_path}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True
    )

    # 设置模型参数
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32  # 默认使用float32
    }

    # 尝试使用量化 (如果指定)
    use_quantization = args.quantization != "None"
    if use_quantization:
        try:
            # 尝试导入必要的库
            import bitsandbytes as bnb
            import accelerate

            logger.info(f"使用 {args.quantization} 量化加载模型")
            if args.quantization == "8bit":
                model_kwargs["load_in_8bit"] = True
                del model_kwargs["torch_dtype"]  # 移除默认的dtype
            elif args.quantization == "4bit":
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_use_double_quant"] = True
                model_kwargs["bnb_4bit_quant_type"] = "nf4"
                del model_kwargs["torch_dtype"]  # 移除默认的dtype
        except (ImportError, Exception) as e:
            logger.warning(f"量化设置失败: {e}")
            logger.warning("将使用默认的非量化模式")
            # 保留默认的torch_dtype=torch.float32

    # 加载基础模型
    try:
        base_model = AutoModel.from_pretrained(
            args.base_model_path,
            **model_kwargs
        )
        logger.info("基础模型加载成功!")
    except Exception as e:
        logger.error(f"加载基础模型时出错: {e}")

        # 如果使用量化失败，尝试非量化模式
        if use_quantization:
            logger.warning("尝试使用非量化模式重新加载...")
            try:
                base_model = AutoModel.from_pretrained(
                    args.base_model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                logger.info("使用非量化模式成功加载基础模型!")
            except Exception as e2:
                logger.error(f"非量化模式加载也失败: {e2}")
                raise
        else:
            raise

    # 加载LoRA模型
    logger.info(f"加载LoRA模型: {args.model_path}")
    model = PeftModel.from_pretrained(
        base_model,
        args.model_path
    )
    logger.info("LoRA模型加载成功!")

    return model, tokenizer

def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    logger.info("=" * 50)
    logger.info("ChatGLM 模型测试脚本")
    logger.info("=" * 50)
    logger.info(f"基础模型: {args.base_model_path}")
    logger.info(f"LoRA模型: {args.model_path}")
    logger.info(f"量化级别: {args.quantization}")
    logger.info("=" * 50)

    # 加载模型和分词器
    model, tokenizer = load_model(args)

    # 设置模型为评估模式
    model.eval()

    # 生成回复
    logger.info("开始生成回复...")
    logger.info(f"提示: {args.prompt}")

    try:
        response, history = model.chat(
            tokenizer,
            args.prompt,
            history=[],
            max_length=args.max_length
        )

        print("\n" + "=" * 50)
        print("生成的回复:")
        print("-" * 50)
        print(response)
        print("=" * 50)
    except Exception as e:
        logger.error(f"生成回复时出错: {e}")

        # 尝试备用生成方法
        logger.warning("尝试使用备用生成方法...")
        try:
            inputs = tokenizer(args.prompt, return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"],
                max_length=args.max_length,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print("\n" + "=" * 50)
            print("生成的回复 (备用方法):")
            print("-" * 50)
            print(response)
            print("=" * 50)
        except Exception as e2:
            logger.error(f"备用生成方法也失败: {e2}")
            raise

if __name__ == "__main__":
    main()