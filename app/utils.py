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
import hashlib
import json
from pathlib import Path
import traceback
from datetime import datetime

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

def get_model_cache_path(model_path, quantization="None", cache_dir="/app/models/cache"):
    """
    根据模型路径和量化类型计算本地缓存路径

    参数:
        model_path: 模型路径或名称
        quantization: 量化类型 ("4bit", "8bit", "None")
        cache_dir: 缓存目录

    返回:
        本地缓存路径
    """
    # 创建缓存目录
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 计算缓存文件名
    model_id = model_path.replace("/", "_")
    cache_key = f"{model_id}_{quantization}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

    return cache_dir / f"{cache_hash}.json"

def check_transformers_version():
    """检查transformers版本兼容性"""
    logger = logging.getLogger(__name__)
    try:
        import transformers
        logger.info(f"transformers版本: {transformers.__version__}")

        if transformers.__version__ < "4.30.0":
            logger.warning(f"当前transformers版本({transformers.__version__})可能不完全支持指定的量化参数")
            logger.warning("推荐版本为 4.30.0 或更高")
    except ImportError:
        logger.warning("无法导入transformers库")

def load_tokenizer(model_path):
    """加载分词器

    参数:
        model_path: 模型路径或名称

    返回:
        tokenizer: 加载的分词器
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载分词器: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 确保分词器有正确的特殊标记
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("分词器加载成功")
        return tokenizer
    except Exception as e:
        logger.error(f"加载分词器失败: {e}")
        logger.error(traceback.format_exc())
        raise

def save_cache_info(model_path, quantization, cache_path):
    """保存模型缓存信息

    参数:
        model_path: 模型路径或名称
        quantization: 量化类型
        cache_path: 缓存文件路径
    """
    logger = logging.getLogger(__name__)

    try:
        # 展开~路径
        cache_dir = os.path.expanduser(os.environ.get("TRANSFORMERS_CACHE", "~/.cache/huggingface"))
        cache_info = {
            "model_path": model_path,
            "quantization": quantization,
            "timestamp": datetime.now().isoformat(),
            "local_path": os.path.join(cache_dir, "models--" + model_path.replace("/", "--"))
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_info, f, indent=2)
        logger.info(f"已创建模型缓存记录: {cache_path}")
    except Exception as e:
        logger.warning(f"创建缓存记录失败: {e}")

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

    # 检查transformers版本兼容性
    check_transformers_version()

    # 主动清理内存，确保有足够空间加载模型
    clean_memory()

    # 检查本地模型缓存
    cache_path = get_model_cache_path(model_path, quantization)
    cache_exists = cache_path.exists()

    if cache_exists:
        logger.info(f"发现本地模型缓存记录: {cache_path}")
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_info = json.load(f)
            logger.info(f"上次下载时间: {cache_info.get('timestamp', '未知')}")
            logger.info(f"本地路径: {cache_info.get('local_path', '未知')}")
        except Exception as e:
            logger.warning(f"读取缓存信息失败: {e}")
            cache_exists = False
    else:
        logger.info("未找到本地缓存记录，模型将从Hugging Face下载")
        # 检查网络连接
        try:
            import requests
            # 尝试访问HF端点
            timeout = 5
            hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
            response = requests.get(f"{hf_endpoint}", timeout=timeout)
            if response.status_code == 200:
                logger.info(f"HF端点连接正常: {hf_endpoint}")
            else:
                logger.warning(f"HF端点响应异常，状态码: {response.status_code}")
        except Exception as e:
            logger.warning(f"检查HF端点连接失败: {e}")
            logger.info("将继续尝试下载模型，但可能遇到网络问题")

    # 加载分词器
    tokenizer = load_tokenizer(model_path)

    # 使用量化模块加载模型
    logger.info("加载模型...")
    model = load_quantized_model(AutoModel, model_path, quantization)

    # 保存缓存信息
    if not cache_exists:
        save_cache_info(model_path, quantization, cache_path)

    logger.info("模型和分词器加载完成!")

    return model, tokenizer

def clean_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def detect_csv_delimiter(file_path):
    """
    检测CSV文件的分隔符

    参数:
        file_path: CSV文件路径

    返回:
        检测到的分隔符
    """
    logger = logging.getLogger(__name__)
    # 默认分隔符
    default_delimiter = ','

    try:
        # 读取前两行进行分析
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip() if f.readline() else ""

        if not first_line:
            logger.warning("文件可能为空")
            return default_delimiter

        # 可能的分隔符及其出现频率
        delimiters = {',': 0, '\t': 0, ';': 0, '|': 0}

        # 检查第一行中各分隔符的数量
        for delimiter in delimiters:
            delimiters[delimiter] = first_line.count(delimiter)

        # 如果第二行存在，也检查它
        if second_line:
            for delimiter in delimiters:
                # 确保两行中的分隔符数量一致
                if second_line.count(delimiter) != delimiters[delimiter] and delimiters[delimiter] > 0:
                    # 数量不一致，可能不是此分隔符
                    delimiters[delimiter] = -1

        # 找出出现次数最多且大于0的分隔符
        max_count = 0
        detected_delimiter = default_delimiter

        for delimiter, count in delimiters.items():
            if count > max_count:
                max_count = count
                detected_delimiter = delimiter

        if max_count > 0:
            logger.info(f"检测到CSV分隔符: '{detected_delimiter}'")
            return detected_delimiter
        else:
            logger.warning(f"无法确定分隔符，使用默认分隔符: '{default_delimiter}'")
            return default_delimiter

    except Exception as e:
        logger.error(f"分隔符检测出错: {e}")
        return default_delimiter

def load_dataset_from_file(dataset_path, text_column):
    """从文件加载数据集

    参数:
        dataset_path: 数据集文件路径
        text_column: 文本列名称

    返回:
        pandas.DataFrame: 加载的数据集
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载数据集: {dataset_path}")

    file_ext = os.path.splitext(dataset_path)[1].lower()

    try:
        if file_ext == '.csv':
            # 自动检测分隔符
            delimiter = detect_csv_delimiter(dataset_path)
            logger.info(f"使用分隔符 '{delimiter}' 加载CSV文件")

            # 尝试自动检测编码
            encodings = ['utf-8', 'gbk', 'latin1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(dataset_path, encoding=encoding, delimiter=delimiter)
                    logger.info(f"成功使用 {encoding} 编码加载CSV")
                    break
                except UnicodeDecodeError:
                    logger.warning(f"{encoding} 编码加载失败，尝试其他编码")
                except Exception as e:
                    logger.error(f"使用 {encoding} 编码加载CSV出错: {e}")
                    raise
            else:
                # 所有编码都失败
                logger.error("无法使用任何已知编码加载CSV")
                raise ValueError("CSV加载失败：编码问题")

        elif file_ext == '.json' or file_ext == '.jsonl':
            # 尝试自动检测是否为jsonl格式
            with open(dataset_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()

            # 检查第一行是否是一个完整的JSON对象
            try:
                json.loads(first_line)
                is_jsonl = True
                logger.info("检测到JSONL格式")
            except json.JSONDecodeError:
                # 不是一个完整的JSON对象，可能是普通JSON
                is_jsonl = file_ext == '.jsonl'

            df = pd.read_json(dataset_path, lines=is_jsonl, encoding='utf-8')

        elif file_ext == '.txt':
            # 简单文本文件，一行一个样本
            with open(dataset_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

            if not texts:
                logger.error(f"数据集文件为空: {dataset_path}")
                raise ValueError("数据集文件不包含任何有效文本")

            df = pd.DataFrame({text_column: texts})
        else:
            error_msg = f"不支持的文件类型: {file_ext}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    except Exception as e:
        logger.error(f"读取数据集出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    # 检查数据是否为空
    if df.empty:
        logger.error("数据集为空")
        raise ValueError("数据集为空，无法继续处理")

    # 检查数据集大小
    logger.info(f"加载的数据集包含 {len(df)} 行, {len(df.columns)} 列")

    # 数据集太大时提示
    if len(df) > 10000:
        logger.warning(f"数据集较大 ({len(df)}行)，可能需要较长处理时间和更多内存")

    # 检查数据集质量
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            logger.warning(f"列 '{col}' 包含 {null_count} 个空值 ({null_count/len(df):.2%})")

    return df

def check_and_fix_text_column(df, text_column):
    """检查文本列并尝试自动修复

    参数:
        df: 数据集DataFrame
        text_column: 文本列名称

    返回:
        适用的文本列名称
    """
    logger = logging.getLogger(__name__)

    # 检查文本列是否存在
    if text_column not in df.columns:
        logger.warning(f"文本列 '{text_column}' 不存在于数据集中。可用列: {', '.join(df.columns)}")

        # 尝试查找可能的文本列
        text_column_candidates = ['text', 'content', 'data', 'input', 'prompt', 'message', 'instruction']

        # 首先尝试精确匹配
        for candidate in text_column_candidates:
            if candidate in df.columns:
                text_column = candidate
                logger.info(f"使用找到的文本列: '{text_column}'")
                return text_column

        # 然后尝试部分匹配
        for candidate in text_column_candidates:
            for col in df.columns:
                if candidate in col.lower():
                    text_column = col
                    logger.info(f"使用部分匹配的文本列: '{text_column}'")
                    return text_column

        # 如果只有一列，自动使用那一列
        if len(df.columns) == 1:
            text_column = df.columns[0]
            logger.info(f"自动使用唯一可用列: '{text_column}'")
            return text_column

        # 尝试寻找字符串类型的列
        string_columns = df.select_dtypes(include=['object']).columns.tolist()
        if string_columns:
            text_column = string_columns[0]
            logger.info(f"自动使用第一个字符串类型列: '{text_column}'")
            return text_column

        # 如果还是找不到
        raise ValueError(f"文本列 '{text_column}' 不存在，无法自动确定替代列")

    # 检查文本列数据类型
    if not pd.api.types.is_string_dtype(df[text_column]):
        logger.warning(f"文本列 '{text_column}' 不是字符串类型，尝试转换...")
        df[text_column] = df[text_column].astype(str)

    return text_column

def tokenize_dataset(dataset, tokenizer, text_column, max_seq_length):
    """对数据集进行分词处理

    参数:
        dataset: Hugging Face数据集
        tokenizer: 分词器
        text_column: 文本列名称
        max_seq_length: 最大序列长度

    返回:
        tokenized_dataset: 分词后的数据集
    """
    logger = logging.getLogger(__name__)

    def tokenize_function(examples):
        texts = examples[text_column]

        # 记录一些样本的长度分布
        if isinstance(texts, list) and len(texts) > 0:
            lengths = [len(text) for text in texts[:100]]  # 只取前100个样本
            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)
            logger.info(f"文本长度分布: 最小={min_length}, 平均={avg_length:.2f}, 最大={max_length}")

            # 检查是否有过短的文本
            short_texts = [text for text in texts[:100] if len(text) < 10]
            if short_texts:
                logger.warning(f"发现 {len(short_texts)} 个过短的文本样本 (长度<10)，可能影响训练质量")
                logger.info(f"样例: {short_texts[:3]}")

        # 分词并截断
        try:
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
        except Exception as e:
            logger.error(f"分词过程中出错: {str(e)}")
            logger.error(f"错误样本: {texts[:100] if isinstance(texts, str) else [t[:100] for t in texts[:3]]}")
            raise

    # 处理数据集
    logger.info(f"对数据集进行分词处理 (最大长度: {max_seq_length})...")
    try:
        # 打印分词器信息
        logger.info(f"分词器类型: {type(tokenizer).__name__}")

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=1,  # CPU环境使用单线程
            remove_columns=dataset.column_names
        )
    except Exception as e:
        logger.error(f"数据集处理出错: {str(e)}")
        logger.error(traceback.format_exc())
        # 创建一个示例，检测是否能正常分词
        logger.info("尝试对单个样本进行分词测试...")
        try:
            sample_text = dataset[0][text_column]
            logger.info(f"样本文本: {sample_text[:100]}...")
            sample_tokens = tokenizer(
                sample_text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt"
            )
            logger.info(f"样本分词成功，形状: {sample_tokens['input_ids'].shape}")

            # 尝试分批处理数据集
            logger.info("尝试使用较小批处理大小...")
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=10,  # 减小批处理大小
                num_proc=1,
                remove_columns=dataset.column_names
            )
        except Exception as sample_error:
            logger.error(f"样本分词也失败: {str(sample_error)}")
            raise

    # 检查分词结果
    try:
        if len(tokenized_dataset) > 0:
            sample_item = tokenized_dataset[0]
            input_shape = sample_item['input_ids'].shape
            logger.info(f"分词后单个样本形状: {input_shape}")
    except Exception as e:
        logger.error(f"检查分词结果出错: {e}")

    return tokenized_dataset

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

    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        logger.error(f"数据集文件不存在: {dataset_path}")
        raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")

    # 加载大型数据集前主动清理内存
    clean_memory()

    # 加载数据集
    df = load_dataset_from_file(dataset_path, text_column)

    # 检查并修复文本列
    text_column = check_and_fix_text_column(df, text_column)

    # 转换为Hugging Face数据集格式
    dataset = Dataset.from_pandas(df)

    # 释放原始数据帧内存
    del df
    gc.collect()

    # 限制样本数量
    if max_samples and max_samples < len(dataset):
        logger.info(f"将数据集限制为 {max_samples} 个样本")
        dataset = dataset.select(range(max_samples))

    logger.info(f"加载了 {len(dataset)} 个样本")

    # 对数据集进行分词处理
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, text_column, max_seq_length)

    # 再次清理内存
    del dataset
    gc.collect()

    # 检查处理后的数据集
    if len(tokenized_dataset) == 0:
        logger.error("处理后的数据集为空")
        raise ValueError("处理后数据集为空，无法继续训练")

    tokenized_dataset.set_format("torch")
    logger.info("数据集准备完成")

    return tokenized_dataset