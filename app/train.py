"""
ChatGLM 训练脚本 - 专为Docker容器优化版本
"""

import os
import argparse
import torch
import gc
from transformers import set_seed
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
import platform

# 导入工具函数
from app.utils import setup_logging, load_model_and_tokenizer, prepare_dataset

# 设置日志
logger = setup_logging(log_file="/app/data/output/train.log")

# 禁用CUDA，仅使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ChatGLM训练脚本 (Docker优化版)")

    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm2-6b",
                        help="模型名称或路径")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA注意力维度")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha参数")
    parser.add_argument("--quantization", type=str, default="None",
                        choices=["4bit", "8bit", "None"],
                        help="模型量化类型")

    # 数据参数
    parser.add_argument("--dataset_path", type=str, default="/app/data/input/dataset.txt",
                    help="本地数据集路径，支持.csv, .json, .jsonl, .txt格式")
    parser.add_argument("--text_column", type=str, default="text", help="文本列名称")
    parser.add_argument("--max_seq_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=1000, help="要使用的最大样本数")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="/app/models/chatglm-lora",
                        help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每个设备的训练批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()

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

def train(args, model, tokenized_dataset):
    """训练循环"""
    logger.info("开始训练...")

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
                if (step + 1) % (args.gradient_accumulation_steps * 50) == 0:
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
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "4")

    logger.info("=" * 50)
    logger.info("ChatGLM CPU 训练脚本 (Docker优化版)")
    logger.info("=" * 50)
    logger.info(f"模型: {args.model_name_or_path}")
    logger.info(f"本地数据集: {args.dataset_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"量化级别: {args.quantization}")
    logger.info(f"LoRA参数: r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info("=" * 50)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.quantization)

    # 添加LoRA适配器
    logger.info("添加LoRA适配器...")
    peft_config = create_lora_config(args)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 准备数据集
    tokenized_dataset = prepare_dataset(
        args.dataset_path,
        args.text_column,
        args.max_samples,
        tokenizer,
        args.max_seq_length
    )

    # 训练模型
    model = train(args, model, tokenized_dataset)

    # 保存最终模型
    logger.info(f"保存模型到 {args.output_dir}")
    model.save_pretrained(args.output_dir)  # 只保存LoRA权重
    tokenizer.save_pretrained(args.output_dir)  # 保存分词器

    logger.info("=" * 50)
    logger.info("训练完成!")
    logger.info(f"模型已保存到: {args.output_dir}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()