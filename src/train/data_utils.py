import os
import json
import logging
import random
import torch
from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_training_dataset(
        dataset_name: Optional[str] = None,
        data_path: Optional[str] = None,
        data_files: Optional[Union[str, List[str], Dict[str, str]]] = None,
        split: Optional[str] = None,
        text_column: str = "text",
        max_samples: Optional[int] = None,
        seed: int = 42
) -> Dataset:
    """加载训练数据集"""
    random.seed(seed)

    if dataset_name:
        logger.info(f"从Hugging Face加载数据集: {dataset_name}")
        if split:
            dataset = load_dataset(dataset_name, split=split)
        else:
            dataset = load_dataset(dataset_name)
    elif data_path:
        logger.info(f"从本地路径加载数据集: {data_path}")
        dataset = load_dataset(data_path, data_files=data_files)
    else:
        raise ValueError("必须提供dataset_name或data_path")

    # 确保dataset是Dataset对象
    if isinstance(dataset, DatasetDict):
        logger.info(f"加载的数据集包含以下分割: {list(dataset.keys())}")
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            # 获取第一个分割
            first_key = list(dataset.keys())[0]
            logger.info(f"使用第一个分割作为训练集: {first_key}")
            dataset = dataset[first_key]

    # 确保包含必要的列
    if text_column not in dataset.column_names:
        available_columns = dataset.column_names
        logger.warning(f"文本列 {text_column} 不在数据集中。可用列: {available_columns}")
        # 尝试使用第一个字符串列作为文本列
        for col in available_columns:
            if len(dataset) > 0 and isinstance(dataset[0][col], str):
                text_column = col
                logger.info(f"使用 {text_column} 作为文本列")
                break

    # 限制样本数量
    if max_samples and max_samples < len(dataset):
        logger.info(f"将数据集大小限制为 {max_samples} 个样本")
        dataset = dataset.select(range(max_samples))

    logger.info(f"加载了 {len(dataset)} 个样本")

    return dataset


def prepare_dataset_for_training(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "text",
        max_seq_length: int = 512,
        add_eos_token: bool = True,
        num_proc: int = 4
) -> Dataset:
    """为语言模型训练准备数据集"""
    logger.info("准备数据集进行训练...")

    # 确保分词器有正确的特殊标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # 处理文本
        texts = examples[text_column]
        if add_eos_token:
            # 检查是否使用ChatGLM分词器
            if hasattr(tokenizer, 'tokenizer_type') and tokenizer.tokenizer_type == 'ChatGLMTokenizer':
                # ChatGLM自动添加EOS，不需要额外添加
                pass
            else:
                texts = [t + tokenizer.eos_token for t in texts]

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

    # 对整个数据集应用预处理
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )

    tokenized_dataset.set_format("torch")
    logger.info(f"数据集准备完成。格式化后的列: {tokenized_dataset.column_names}")

    return tokenized_dataset


def create_chat_prompt(messages, system_prompt=None):
    """创建聊天格式的提示词"""
    prompt = ""

    # 添加系统提示词
    if system_prompt:
        prompt += f"<|system|>\n{system_prompt}\n\n"

    # 添加聊天消息
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|{role}|>\n{content}\n\n"

    # 添加最后的助手提示
    prompt += "<|assistant|>\n"

    return prompt


def prepare_instruction_dataset(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        instruction_column: str = "instruction",
        input_column: str = "input",
        output_column: str = "output",
        max_seq_length: int = 512,
        num_proc: int = 4
) -> Dataset:
    """为指令微调准备数据集"""
    logger.info("为指令微调准备数据集...")

    def format_instruction(example):
        instruction = example[instruction_column]

        if input_column in example and example[input_column]:
            input_text = example[input_column]
            text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 响应:\n"
        else:
            text = f"### 指令:\n{instruction}\n\n### 响应:\n"

        # 添加输出文本
        example["text"] = text + example[output_column]
        return example

    # 应用格式化
    formatted_dataset = dataset.map(
        format_instruction,
        num_proc=num_proc
    )

    # 对格式化后的数据集进行分词
    tokenized_dataset = prepare_dataset_for_training(
        formatted_dataset,
        tokenizer,
        text_column="text",
        max_seq_length=max_seq_length,
        add_eos_token=True,
        num_proc=num_proc
    )

    return tokenized_dataset