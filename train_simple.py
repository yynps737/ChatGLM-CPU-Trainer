"""
ChatGLM 简化版训练脚本 - 专为低资源CPU环境设计
"""

import os
import argparse
import logging
import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import platform

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 禁用CUDA，仅使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ChatGLM简化版训练脚本 (CPU)")

    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm2-6b",
                        help="模型名称或路径")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA注意力维度")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha参数")
    parser.add_argument("--quantization", type=str, default="None",
                        choices=["4bit", "8bit", "None"],
                        help="模型量化类型")

    # 数据参数
    parser.add_argument("--dataset_name", type=str, default="uer/cluecorpussmall",
                        help="Hugging Face数据集名称")
    parser.add_argument("--text_column", type=str, default="text", help="文本列名称")
    parser.add_argument("--max_seq_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=1000, help="要使用的最大样本数")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output/chatglm-lora",
                        help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每个设备的训练批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()

def load_model_and_tokenizer(args):
    """加载模型和分词器"""
    logger.info(f"加载模型: {args.model_name_or_path}")

    # 清理内存
    gc.collect()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    # 确保分词器有正确的特殊标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 设置模型参数，默认使用非量化模式
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

    # 加载模型
    try:
        logger.info("开始加载模型...")
        logger.info(f"使用参数: {model_kwargs}")

        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            **model_kwargs
        )
        logger.info("模型加载成功!")
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")

        # 如果使用量化失败，尝试非量化模式
        if use_quantization:
            logger.warning("尝试使用非量化模式重新加载...")
            try:
                model = AutoModel.from_pretrained(
                    args.model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                logger.info("使用非量化模式成功加载模型!")
            except Exception as e2:
                logger.error(f"非量化模式加载也失败: {e2}")
                raise
        else:
            raise

    return model, tokenizer

def create_lora_config(args):
    """创建LoRA配置"""
    # 为ChatGLM模型定制的目标模块
    target_modules = [
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
    ]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )

def prepare_dataset(args, tokenizer):
    """准备数据集"""
    logger.info(f"加载数据集: {args.dataset_name}")

    # 加载数据集
    dataset = load_dataset(args.dataset_name)
    if "train" in dataset:
        dataset = dataset["train"]
    else:
        # 使用第一个可用分割
        first_key = list(dataset.keys())[0]
        dataset = dataset[first_key]

    # 限制样本数量
    if args.max_samples and args.max_samples < len(dataset):
        logger.info(f"将数据集限制为 {args.max_samples} 个样本")
        dataset = dataset.select(range(args.max_samples))

    logger.info(f"加载了 {len(dataset)} 个样本")

    # 检查文本列是否存在
    if args.text_column not in dataset.column_names:
        available_columns = dataset.column_names
        logger.warning(f"文本列 '{args.text_column}' 不在数据集中。可用列: {available_columns}")
        # 尝试使用第一个字符串列
        for col in available_columns:
            if len(dataset) > 0 and isinstance(dataset[0][col], str):
                args.text_column = col
                logger.info(f"使用 '{args.text_column}' 作为文本列")
                break

    # 分词函数
    def tokenize_function(examples):
        texts = examples[args.text_column]
        # ChatGLM自动添加EOS，不需要额外添加

        # 分词并截断
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )

        # 为因果语言模型准备标签
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    # 处理数据集
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,  # CPU环境使用单线程
        remove_columns=dataset.column_names
    )

    tokenized_dataset.set_format("torch")
    logger.info("数据集准备完成")

    return tokenized_dataset

def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置CPU环境变量
    if platform.system() == "Windows":
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)

    logger.info("=" * 50)
    logger.info("ChatGLM CPU 简化版训练脚本")
    logger.info("=" * 50)
    logger.info(f"模型: {args.model_name_or_path}")
    logger.info(f"数据集: {args.dataset_name}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"量化级别: {args.quantization}")
    logger.info(f"LoRA参数: r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info("=" * 50)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args)

    # 添加LoRA适配器
    logger.info("添加LoRA适配器...")
    peft_config = create_lora_config(args)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 准备数据集
    tokenized_dataset = prepare_dataset(args, tokenizer)

    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        # CPU训练特定设置
        fp16=False,  # CPU不支持FP16
        bf16=False,  # CPU不支持BF16
        # 其他设置
        remove_unused_columns=False,
        evaluation_strategy="no",
        save_strategy="steps",
    )

    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 保存模型
    logger.info(f"保存模型到 {args.output_dir}")
    model.save_pretrained(args.output_dir)  # 只保存LoRA权重
    tokenizer.save_pretrained(args.output_dir)  # 保存分词器

    logger.info("=" * 50)
    logger.info("训练完成!")
    logger.info(f"模型已保存到: {args.output_dir}")
    logger.info(f"测试模型命令: python test_model.py --model_path {args.output_dir} --base_model_path {args.model_name_or_path}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()