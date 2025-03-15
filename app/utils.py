"""
ChatGLM-CPU-Trainer 共用工具函数
"""

import os
import gc
import logging
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datasets import Dataset

# 导入量化模块
from app.quantization import load_quantized_model

# 设置日志格式
def setup_logging(log_file=None):
    """设置日志配置"""
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)

def load_model_and_tokenizer(model_path, quantization="None"):
    """加载模型和分词器

    参数:
        model_path: 模型路径或名称
        quantization: 量化类型 ("4bit", "8bit", "None")

    返回:
        model, tokenizer
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载模型: {model_path}")

    # 清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 确保分词器有正确的特殊标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 使用量化模块加载模型
    model = load_quantized_model(AutoModel, model_path, quantization)

    return model, tokenizer

def prepare_dataset(dataset_path, text_column, max_samples, tokenizer, max_seq_length):
    """准备本地数据集

    参数:
        dataset_path: 数据集路径
        text_column: 文本列名称
        max_samples: 最大样本数
        tokenizer: 分词器
        max_seq_length: 最大序列长度

    返回:
        tokenized_dataset
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载本地数据集: {dataset_path}")

    # 根据文件类型加载数据
    file_ext = os.path.splitext(dataset_path)[1].lower()

    if file_ext == '.csv':
        df = pd.read_csv(dataset_path)
    elif file_ext == '.json' or file_ext == '.jsonl':
        df = pd.read_json(dataset_path, lines=file_ext == '.jsonl')
    elif file_ext == '.txt':
        # 简单文本文件，一行一个样本
        with open(dataset_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({text_column: texts})
    else:
        raise ValueError(f"不支持的文件类型: {file_ext}")

    # 转换为Hugging Face数据集格式
    dataset = Dataset.from_pandas(df)

    # 限制样本数量
    if max_samples and max_samples < len(dataset):
        logger.info(f"将数据集限制为 {max_samples} 个样本")
        dataset = dataset.select(range(max_samples))

    logger.info(f"加载了 {len(dataset)} 个样本")

    # 分词函数
    def tokenize_function(examples):
        texts = examples[text_column]

        # 分词并截断
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
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