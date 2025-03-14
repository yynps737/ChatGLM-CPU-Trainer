"""
简化版训练脚本 - 不使用DeepSpeed，适用于Windows和简单环境
"""

import os
import argparse
import logging
import torch
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import psutil

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ChatGLM简化训练脚本")

    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm3-6b", help="模型名称或路径")
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA注意力维度")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA Dropout率")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "8bit", "4bit"], help="模型量化级别")

    # 数据集参数
    parser.add_argument("--dataset_name", type=str, default="uer/cluecorpussmall", help="数据集名称")
    parser.add_argument("--text_column", type=str, default="text", help="文本列名称")
    parser.add_argument("--max_seq_length", type=int, default=256, help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数量")
    parser.add_argument("--instruction_format", action="store_true", help="是否使用指令格式")
    parser.add_argument("--instruction_column", type=str, default="instruction", help="指令列名称")
    parser.add_argument("--input_column", type=str, default="input", help="输入列名称")
    parser.add_argument("--output_column", type=str, default="output", help="输出列名称")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output/chatglm3-lora", help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的训练批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


def is_chatglm_model(model_name_or_path):
    """判断是否为ChatGLM模型"""
    return "chatglm" in model_name_or_path.lower() or "glm" in model_name_or_path.lower()


def load_model_and_tokenizer(args):
    """加载模型和分词器"""
    logger.info(f"加载模型: {args.model_name_or_path}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    # 确保分词器有正确的特殊标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型的参数
    load_in_8bit = args.quantization == "8bit"
    load_in_4bit = args.quantization == "4bit"

    model_kwargs = {
        "trust_remote_code": True,
    }

    # 应用量化设置
    try:
        if load_in_8bit:
            logger.info("使用8位量化加载模型")
            import bitsandbytes as bnb
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            logger.info("使用4位量化加载模型")
            import bitsandbytes as bnb
            model_kwargs["load_in_4bit"] = True
        else:
            logger.info("使用32位精度加载模型")
            model_kwargs["torch_dtype"] = torch.float32
    except ImportError:
        logger.warning("无法导入bitsandbytes，将使用32位精度加载模型")
        model_kwargs["torch_dtype"] = torch.float32

    # 根据模型类型加载
    is_glm = is_chatglm_model(args.model_name_or_path)
    try:
        if is_glm:
            logger.info("检测到ChatGLM模型，使用专用加载方式")
            model = AutoModel.from_pretrained(
                args.model_name_or_path,
                **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                **model_kwargs
            )
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        logger.info("尝试不带量化加载...")
        if is_glm:
            model = AutoModel.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )

    return model, tokenizer


def load_and_process_dataset(args, tokenizer):
    """加载和处理数据集"""
    logger.info(f"加载数据集: {args.dataset_name}")

    # 加载数据集
    try:
        dataset = load_dataset(args.dataset_name)
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            # 获取第一个分割
            first_key = list(dataset.keys())[0]
            logger.info(f"使用分割: {first_key}")
            dataset = dataset[first_key]
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise

    # 限制样本数量
    if args.max_samples and args.max_samples < len(dataset):
        logger.info(f"限制数据集大小为 {args.max_samples} 个样本")
        dataset = dataset.select(range(args.max_samples))

    logger.info(f"加载了 {len(dataset)} 个样本")

    # 确保包含必要的列
    if args.text_column not in dataset.column_names:
        available_columns = dataset.column_names
        logger.warning(f"文本列 {args.text_column} 不在数据集中。可用列: {available_columns}")
        # 尝试使用第一个字符串列作为文本列
        for col in available_columns:
            if len(dataset) > 0 and isinstance(dataset[0][col], str):
                args.text_column = col
                logger.info(f"使用 {args.text_column} 作为文本列")
                break

    # 分词处理函数
    def tokenize_function(examples):
        # 处理文本
        if args.instruction_format:
            # 使用指令格式
            texts = []
            for i in range(len(examples[args.instruction_column])):
                instruction = examples[args.instruction_column][i]

                if args.input_column in examples and examples[args.input_column][i]:
                    input_text = examples[args.input_column][i]
                    text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 响应:\n"
                else:
                    text = f"### 指令:\n{instruction}\n\n### 响应:\n"

                # 添加输出文本
                text += examples[args.output_column][i]
                texts.append(text)
        else:
            # 使用标准格式
            texts = examples[args.text_column]

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

    # 对整个数据集应用预处理
    try:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        tokenized_dataset.set_format("torch")
    except Exception as e:
        logger.error(f"处理数据集失败: {e}")
        raise

    logger.info(f"数据集处理完成。格式化后的列: {tokenized_dataset.column_names}")
    return tokenized_dataset


def add_lora_to_model(model, args):
    """将LoRA适配器添加到模型"""
    if not args.use_lora:
        return model

    logger.info("添加LoRA适配器...")
    try:
        # 为ChatGLM定制的目标模块
        target_modules = [
            "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
        ]

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except Exception as e:
        logger.error(f"添加LoRA失败: {e}")
        logger.warning("将使用完整微调")

    return model


def create_trainer(model, tokenizer, train_dataset, args):
    """创建Trainer"""
    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        fp16=False,  # 在CPU上禁用fp16
        remove_unused_columns=False,  # 需要为LM任务保留列
    )

    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 确保使用CPU，禁用CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # 检查内存状态
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    logger.info(f"系统可用内存: {available_memory:.2f} GB")

    # 内存不足时自动调整
    if available_memory < 8 and args.quantization == "8bit":
        logger.warning(f"可用内存不足8GB，自动切换到4bit量化")
        args.quantization = "4bit"

    if available_memory < 5 and args.max_samples is None:
        max_samples = int(min(5000, 2000 + (available_memory * 1000)))
        logger.warning(f"可用内存有限，自动限制训练样本数为: {max_samples}")
        args.max_samples = max_samples

    # 处理输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型和分词器
    try:
        model, tokenizer = load_model_and_tokenizer(args)
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return

    # 加载和处理数据集
    try:
        train_dataset = load_and_process_dataset(args, tokenizer)
    except Exception as e:
        logger.error(f"准备数据集失败: {e}")
        return

    # 添加LoRA (如果需要)
    if args.use_lora:
        model = add_lora_to_model(model, args)

    # 创建Trainer
    trainer = create_trainer(model, tokenizer, train_dataset, args)

    # 训练模型
    logger.info("开始训练...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        return

    # 保存模型
    logger.info(f"保存模型到 {args.output_dir}")
    try:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        logger.error(f"保存模型出错: {e}")

    logger.info("训练完成！")


if __name__ == "__main__":
    main()