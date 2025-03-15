"""
ChatGLM 简化版训练脚本 - 专为低资源CPU环境设计 (兼容版)
"""

import os
import argparse
import logging
import torch
import gc
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import platform

# 完全避免使用Trainer类，使用简化的训练循环
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

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
    parser.add_argument("--dataset_name", type=str, default=None,
                    help="Hugging Face数据集名称")
    parser.add_argument("--dataset_path", type=str, default=None,
                    help="本地数据集路径，支持.csv, .json, .jsonl, .txt格式")
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

def prepare_local_dataset(args, tokenizer):
    """准备本地数据集"""
    logger.info(f"加载本地数据集: {args.dataset_path}")

    # 根据文件类型加载数据
    file_path = args.dataset_path
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path, lines=True if file_path.endswith('.jsonl') else False)
    elif file_path.endswith('.txt'):
        # 简单文本文件，一行一个样本
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({args.text_column: texts})
    else:
        raise ValueError(f"不支持的文件类型: {file_path}")

    # 转换为Hugging Face数据集格式
    dataset = Dataset.from_pandas(df)

    # 限制样本数量
    if args.max_samples and args.max_samples < len(dataset):
        logger.info(f"将数据集限制为 {args.max_samples} 个样本")
        dataset = dataset.select(range(args.max_samples))

    logger.info(f"加载了 {len(dataset)} 个样本")

    # 分词函数
    def tokenize_function(examples):
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

def train(args, model, tokenized_dataset):
    """简化的训练循环，避免使用Trainer"""
    logger.info("开始手动训练循环...")

    # 设置数据加载器
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True
    )

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 开始训练循环
    model.train()

    for epoch in range(args.num_train_epochs):
        logger.info(f"开始训练第 {epoch+1}/{args.num_train_epochs} 轮")
        running_loss = 0.0

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        accumulated_loss = 0

        for step, batch in enumerate(progress_bar):
            # 将数据移动到设备上
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()

            # 梯度累积
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                running_loss += accumulated_loss
                progress_bar.set_postfix({'loss': accumulated_loss * args.gradient_accumulation_steps})
                accumulated_loss = 0

                # 检查点保存
                if (step + 1) % (args.gradient_accumulation_steps * 200) == 0:
                    logger.info(f"保存检查点 epoch {epoch+1}, step {step+1}")
                    save_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}-step{step+1}")
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)

        # 保存每个epoch的模型
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)

        logger.info(f"第 {epoch+1} 轮结束, 平均损失: {running_loss / len(data_loader)}")

    return model

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
    if args.dataset_name:
        logger.info(f"数据集: {args.dataset_name}")
    if args.dataset_path:
        logger.info(f"本地数据集: {args.dataset_path}")
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

    # 准备本地数据集
    tokenized_dataset = prepare_local_dataset(args, tokenizer)

    # 手动训练循环，避免使用Trainer
    model = train(args, model, tokenized_dataset)

    # 保存最终模型
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