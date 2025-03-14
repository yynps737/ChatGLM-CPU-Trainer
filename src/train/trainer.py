import os
import logging
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from typing import Dict, List, Union, Optional, Any

logger = logging.getLogger(__name__)


def create_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        args=None,
        data_collator=None
):

    if not torch.cuda.is_available():
        logger.info("检测到CPU训练环境，应用CPU特定优化...")
        # 禁用混合精度训练
        if hasattr(args, 'fp16') and args.fp16:
            logger.warning("在CPU上不支持FP16训练，已自动禁用")
            args.fp16 = False
        if hasattr(args, 'bf16') and args.bf16:
            logger.warning("在CPU上不支持BF16训练，已自动禁用")
            args.bf16 = False

    """创建训练器

    Args:
        model: 要训练的模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        args: 训练参数
        data_collator: 数据整理器

    Returns:
        trainer: Trainer对象
    """
    # 如果没有提供data_collator，创建默认的
    if data_collator is None:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer


def create_training_args(
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        logging_steps: int = 10,
        save_steps: int = 500,
        save_total_limit: int = 3,
        fp16: bool = False,
        bf16: bool = False,
        deepspeed: Optional[str] = None,
        **kwargs
) -> TrainingArguments:
    # 检查CPU环境并强制禁用混合精度
    if not torch.cuda.is_available():
        if fp16 or bf16:
            logger.warning("在CPU环境中，已自动禁用FP16和BF16设置")
        fp16 = False
        bf16 = False


    """创建训练参数

    Args:
        output_dir: 输出目录
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 每个设备的训练批大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        weight_decay: 权重衰减
        warmup_ratio: 预热比例
        logging_steps: 日志记录步数
        save_steps: 保存步数
        save_total_limit: 保存的检查点数量限制
        fp16: 是否使用fp16
        bf16: 是否使用bf16
        deepspeed: DeepSpeed配置文件路径
        **kwargs: 其他参数

    Returns:
        TrainingArguments对象
    """
    # 创建基本的训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        fp16=fp16,
        bf16=bf16,
        deepspeed=deepspeed,
        # 默认设置
        evaluation_strategy="no",
        save_strategy="steps",
        optim="adamw_torch",
        # 其他设置
        remove_unused_columns=False,  # 需要为LM任务保留列
        **kwargs
    )

    return training_args


def get_last_checkpoint(output_dir: str) -> Optional[str]:
    """获取最新的检查点

    Args:
        output_dir: 输出目录

    Returns:
        最新检查点的路径，如果没有则返回None
    """
    if not os.path.exists(output_dir):
        return None

    checkpoints = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-")
    ]

    if not checkpoints:
        return None

    # 按照检查点号排序
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = os.path.join(output_dir, checkpoints[-1])

    logger.info(f"找到最新检查点: {latest_checkpoint}")
    return latest_checkpoint