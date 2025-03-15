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

    # 设置模型参数，默认使用非量化模式
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32  # 默认使用float32
    }

    # 尝试使用量化 (如果指定)
    use_quantization = quantization != "None"
    if use_quantization:
        try:
            # 尝试导入必要的库
            import bitsandbytes as bnb
            from accelerate import init_empty_weights

            logger.info(f"使用 {quantization} 量化加载模型")
            logger.info(f"bitsandbytes版本: {bnb.__version__}")

            if quantization == "8bit":
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = {"": "cpu"}
                del model_kwargs["torch_dtype"]  # 移除默认的dtype
            elif quantization == "4bit":
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_use_double_quant"] = True
                model_kwargs["bnb_4bit_quant_type"] = "nf4"
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float32
                model_kwargs["device_map"] = {"": "cpu"}
                del model_kwargs["torch_dtype"]  # 移除默认的dtype

        except (ImportError, Exception) as e:
            logger.warning(f"量化设置失败 (错误详情: {str(e)})")
            logger.warning("将使用默认的非量化模式")
            use_quantization = False
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float32
            }

    # 加载模型
    try:
        logger.info("开始加载模型...")
        logger.info(f"使用参数: {model_kwargs}")

        model = AutoModel.from_pretrained(
            model_path,
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
                    model_path,
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