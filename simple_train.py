"""
简化版训练脚本 - 专为低资源环境设计的ChatGLM训练工具
不依赖DeepSpeed，使用基本的PyTorch训练循环
"""

import os
import argparse
import logging
import torch
import psutil
import gc
import platform
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model
import sys

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
    parser = argparse.ArgumentParser(description="简化版ChatGLM训练脚本 (CPU)")

    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm3-6b",
                        help="模型名称或路径")
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA注意力维度")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA的Dropout率")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "8bit", "4bit"],
                        help="模型量化类型 (8bit, 4bit 或 不量化)")

    # 数据参数
    parser.add_argument("--dataset_name", type=str, default="uer/cluecorpussmall",
                        help="Hugging Face数据集名称")
    parser.add_argument("--data_path", type=str, default=None, help="本地数据集路径")
    parser.add_argument("--text_column", type=str, default="text", help="文本列名称")
    parser.add_argument("--max_seq_length", type=int, default=256, help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=None, help="要使用的最大样本数")
    parser.add_argument("--instruction_format", action="store_true", help="是否使用指令格式")
    parser.add_argument("--instruction_column", type=str, default="instruction", help="指令列名称")
    parser.add_argument("--input_column", type=str, default="input", help="输入列名称")
    parser.add_argument("--output_column", type=str, default="output", help="输出列名称")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每个设备的训练批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="保存的检查点数量限制")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


def is_chatglm_model(model_name_or_path):
    """判断是否为ChatGLM模型"""
    return "chatglm" in model_name_or_path.lower() or "glm" in model_name_or_path.lower()


def load_model_and_tokenizer(args):
    """加载模型和分词器 - 针对低内存环境优化"""
    logger.info(f"加载模型: {args.model_name_or_path}")

    # 在加载模型之前先清理内存
    gc.collect()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    # 确保分词器有正确的特殊标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检查可用内存，并提供警告
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    model_size_estimate = 6.0  # 假设模型大约为6GB (ChatGLM2-6B非量化)
    if "chatglm3" in args.model_name_or_path.lower():
        model_size_estimate = 12.0  # ChatGLM3-6B更大

    # 如果是ChatGLM3但内存少于8GB，自动切换到ChatGLM2
    if "chatglm3" in args.model_name_or_path.lower() and available_memory < 8.0 and args.quantization != "4bit":
        logger.warning(f"内存不足 ({available_memory:.1f}GB) 加载 ChatGLM3-6B 模型!")
        logger.warning(f"推荐: 1) 使用4bit量化 2) 切换到ChatGLM2-6B 3) 增加系统内存")

    # 加载模型的参数
    load_in_8bit = args.quantization == "8bit"
    load_in_4bit = args.quantization == "4bit"

    model_kwargs = {
        "trust_remote_code": True,
    }

    # 尝试启用量化
    try:
        if load_in_8bit or load_in_4bit:
            try:
                import bitsandbytes as bnb
                if load_in_8bit:
                    logger.info("使用8位量化加载模型")
                    model_kwargs["load_in_8bit"] = True
                else:
                    logger.info("使用4位量化加载模型 (超低内存模式)")
                    model_kwargs["load_in_4bit"] = True
                    # 启用Double量化以进一步减少内存使用
                    model_kwargs["bnb_4bit_use_double_quant"] = True
                    model_kwargs["bnb_4bit_quant_type"] = "nf4"
                    model_kwargs["bnb_4bit_compute_dtype"] = torch.float32
            except ImportError:
                logger.warning(f"bitsandbytes导入失败，尝试替代方法")
                # 尝试自动解决bitsandbytes问题
                if "Windows" in platform.system():
                    try:
                        logger.warning("尝试自动安装Windows兼容的bitsandbytes...")
                        # 尝试卸载现有版本
                        os.system("pip uninstall -y bitsandbytes-windows")
                        # 尝试安装Windows兼容版本
                        os.system("pip install bitsandbytes-windows")
                        import bitsandbytes as bnb
                        logger.info("成功安装bitsandbytes-windows")
                        if load_in_4bit:
                            model_kwargs["load_in_4bit"] = True
                        else:
                            model_kwargs["load_in_8bit"] = True
                    except:
                        logger.warning("自动安装失败，将使用非量化模式 (警告: 可能导致内存不足)")
                        model_kwargs["torch_dtype"] = torch.float32
                else:
                    model_kwargs["torch_dtype"] = torch.float32
        else:
            logger.info("使用32位精度加载模型 (警告: 内存使用量大)")
            model_kwargs["torch_dtype"] = torch.float32
    except Exception as e:
        logger.warning(f"量化设置失败: {e}")
        logger.warning("回退到32位精度 (警告: 内存使用量大)")
        model_kwargs["torch_dtype"] = torch.float32

    # 根据模型类型加载
    is_glm = is_chatglm_model(args.model_name_or_path)

    try:
        # 尝试逐步加载，以更好地处理低内存情况
        logger.info("开始加载模型...")

        # 确保清理任何已缓存的模型
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

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

        logger.info("模型加载成功!")

    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "DefaultCPUAllocator: can't allocate memory" in str(e):
            logger.error(f"内存不足错误: {e}")
            logger.error("请尝试: 1) 减少序列长度和样本数 2) 使用4bit量化 3) 使用更小的模型")
            raise RuntimeError("内存不足，无法加载模型。请参考上面的建议。")
        elif "bitsandbytes" in str(e):
            logger.error(f"BitsAndBytes错误: {e}")
            logger.error("尝试非量化模式...")
            # 尝试非量化模式
            try:
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
                logger.info("使用非量化模式成功加载模型")
            except Exception as e2:
                logger.error(f"非量化模式加载失败: {e2}")
                raise
        else:
            logger.error(f"加载模型时出错: {e}")
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
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )


def load_training_dataset(args):
    """加载训练数据集"""
    if args.dataset_name:
        logger.info(f"从Hugging Face加载数据集: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name)
    elif args.data_path:
        logger.info(f"从本地路径加载数据集: {args.data_path}")
        dataset = load_dataset(args.data_path)
    else:
        raise ValueError("必须提供dataset_name或data_path")

    # 确保dataset是Dataset对象
    if hasattr(dataset, 'train'):
        dataset = dataset['train']
    elif len(dataset) > 0 and not isinstance(dataset, Dataset):
        # 使用第一个分割
        first_key = list(dataset.keys())[0]
        logger.info(f"使用分割 '{first_key}' 作为训练集")
        dataset = dataset[first_key]

    # 确保包含必要的列
    if args.text_column not in dataset.column_names:
        available_columns = dataset.column_names
        logger.warning(f"文本列 '{args.text_column}' 不在数据集中。可用列: {available_columns}")
        # 尝试使用第一个字符串列作为文本列
        for col in available_columns:
            if len(dataset) > 0 and isinstance(dataset[0][col], str):
                args.text_column = col
                logger.info(f"使用 '{args.text_column}' 作为文本列")
                break

    # 限制样本数量
    if args.max_samples and args.max_samples < len(dataset):
        logger.info(f"将数据集大小限制为 {args.max_samples} 个样本")
        dataset = dataset.select(range(args.max_samples))

    logger.info(f"加载了 {len(dataset)} 个样本")
    return dataset


def prepare_dataset(args, dataset, tokenizer):
    """准备数据集"""
    logger.info("准备数据集...")

    # 处理指令格式数据集
    if args.instruction_format:
        logger.info("使用指令格式...")

        def format_instruction(example):
            instruction = example[args.instruction_column]

            if args.input_column in example and example[args.input_column]:
                input_text = example[args.input_column]
                text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 响应:\n"
            else:
                text = f"### 指令:\n{instruction}\n\n### 响应:\n"

            # 添加输出文本
            example["text"] = text + example[args.output_column]
            return example

        # 应用格式化
        dataset = dataset.map(format_instruction)
        text_column = "text"
    else:
        text_column = args.text_column

    # 分词函数
    def tokenize_function(examples):
        texts = examples[text_column]
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
            max_length=args.max_seq_length,
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
        num_proc=1,  # 对于CPU训练使用单进程
        remove_columns=dataset.column_names
    )

    tokenized_dataset.set_format("torch")
    logger.info(f"数据集准备完成，共有 {len(tokenized_dataset)} 个样本")

    return tokenized_dataset


def train(args):
    """训练函数"""
    # 设置随机种子
    set_seed(args.seed)

    # 优化内存使用
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 检查可用内存
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    logger.info(f"训练开始时系统可用内存: {available_memory:.2f} GB")

    # 内存不足时自动调整量化级别
    if available_memory < 8 and args.quantization == "8bit":
        logger.warning(f"可用内存不足8GB，自动切换到4bit量化")
        args.quantization = "4bit"

    if available_memory < 5 and args.max_samples is None:
        max_samples = int(min(5000, 2000 + (available_memory * 1000)))
        logger.warning(f"可用内存有限，自动限制训练样本数为: {max_samples}")
        args.max_samples = max_samples

    # 加载模型和分词器
    try:
        model, tokenizer = load_model_and_tokenizer(args)
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        if "bitsandbytes" in str(e) or "accelerate" in str(e):
            logger.error("量化依赖缺失，尝试禁用量化并重新加载模型")
            args.quantization = None
            model, tokenizer = load_model_and_tokenizer(args)
        else:
            raise

    # 添加LoRA适配器（如果启用）
    if args.use_lora:
        logger.info("添加LoRA适配器...")
        peft_config = create_lora_config(args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 加载数据集
    dataset = load_training_dataset(args)

    # 准备数据集
    tokenized_dataset = prepare_dataset(args, dataset, tokenizer)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # CPU训练特定设置
        fp16=False,  # CPU不支持FP16
        bf16=False,  # CPU不支持BF16
        # 其他设置
        evaluation_strategy="no",
        save_strategy="steps",
        optim="adamw_torch",
        remove_unused_columns=False,  # 需要为LM任务保留列
    )

    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 使用因果语言模型 (不使用MLM)
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
    try:
        trainer.train()

        # 保存模型
        logger.info(f"保存模型到 {args.output_dir}")
        if args.use_lora:
            # 仅保存LoRA权重
            model.save_pretrained(args.output_dir)
        else:
            # 保存完整模型
            trainer.save_model(args.output_dir)

        # 保存分词器
        tokenizer.save_pretrained(args.output_dir)

        logger.info("训练完成！")
        return True
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        return False


def main():
    """主函数"""
    # 检查Python版本
    if not (sys.version_info.major == 3 and 8 <= sys.version_info.minor <= 13):
        logger.warning(f"警告: Python {sys.version_info.major}.{sys.version_info.minor} 可能不兼容，推荐使用Python 3.8-3.13")

    # 获取命令行参数
    args = parse_args()

    # 确保使用CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    logger.info("=" * 50)
    logger.info("ChatGLM CPU 训练脚本 - 简化版")
    logger.info("=" * 50)
    logger.info(f"模型: {args.model_name_or_path}")
    logger.info(f"数据集: {args.dataset_name or args.data_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"量化级别: {args.quantization or '不量化'}")
    if args.use_lora:
        logger.info(f"使用LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    else:
        logger.info("未使用LoRA (将训练完整模型)")
    logger.info("=" * 50)

    # 开始训练
    success = train(args)

    if success:
        logger.info("=" * 50)
        logger.info("训练成功完成!")
        logger.info(f"模型已保存到: {args.output_dir}")
        if args.use_lora:
            logger.info(f"测试模型命令: python test_model.py --model_path {args.output_dir} --base_model_path {args.model_name_or_path} --is_peft_model --quantization {args.quantization or 'None'}")
        else:
            logger.info(f"测试模型命令: python test_model.py --model_path {args.output_dir}")
        logger.info("=" * 50)
    else:
        logger.error("=" * 50)
        logger.error("训练失败!")
        logger.error("请检查上面的错误信息")
        logger.error("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()