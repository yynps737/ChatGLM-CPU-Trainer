import os
import argparse
import logging
import torch
import psutil
from transformers import set_seed, HfArgumentParser, AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.models.model_utils import load_base_model, create_peft_config, add_lora_to_model, save_model_and_tokenizer
from src.train.data_utils import load_training_dataset, prepare_dataset_for_training, prepare_instruction_dataset
from src.train.trainer import create_trainer, create_training_args, get_last_checkpoint
from src.utils.system_utils import optimize_memory_usage, generate_optimized_ds_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        default="THUDM/chatglm3-6b",
        metadata={"help": "模型名称或路径"}
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "是否使用PEFT"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "是否使用LoRA (将设置use_peft=True)"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA注意力维度"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA Alpha参数"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA的Dropout率"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "要应用LoRA的目标模块"}
    )
    quantization: Optional[str] = field(
        default=None,
        metadata={"help": "模型量化类型 (8bit, 4bit 或 不量化)"}
    )
    max_memory: Optional[Dict[int, str]] = field(
        default=None,
        metadata={"help": "每个GPU的最大内存（以GB为单位）"}
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    dataset_name: Optional[str] = field(
        default="uer/cluecorpussmall",
        metadata={"help": "Hugging Face数据集名称"}
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "本地数据集路径"}
    )
    data_files: Optional[List[str]] = field(
        default=None,
        metadata={"help": "数据文件列表"}
    )
    text_column: str = field(
        default="text",
        metadata={"help": "文本列名称"}
    )
    max_seq_length: int = field(
        default=256,
        metadata={"help": "最大序列长度"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "要使用的最大样本数"}
    )
    instruction_format: bool = field(
        default=False,
        metadata={"help": "是否使用指令格式"}
    )
    instruction_column: str = field(
        default="instruction",
        metadata={"help": "指令列名称"}
    )
    input_column: str = field(
        default="input",
        metadata={"help": "输入列名称"}
    )
    output_column: str = field(
        default="output",
        metadata={"help": "输出列名称"}
    )


@dataclass
class TrainingArguments:
    """训练相关参数"""
    output_dir: str = field(
        default="./output",
        metadata={"help": "输出目录"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "每个设备的训练批大小"}
    )
    gradient_accumulation_steps: int = field(
        default=32,
        metadata={"help": "梯度累积步数"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "学习率"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "权重衰减"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "预热比例"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "日志记录步数"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "保存步数"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "保存的检查点数量限制"}
    )
    auto_optimize_ds_config: bool = field(
        default=False,
        metadata={"help": "是否自动优化DeepSpeed配置"}
    )
    ds_config_path: Optional[str] = field(
        default="configs/default_config.json",
        metadata={"help": "DeepSpeed配置文件路径"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "是否使用FP16"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "是否使用BF16"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "从检查点恢复"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )


def parse_args():
    """解析命令行参数"""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    # 解析命令行参数
    if len(os.sys.argv) > 1 and os.sys.argv[1].endswith(".json"):
        # 如果提供了JSON文件，使用它
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(os.sys.argv[1])
        )
    else:
        # 否则使用命令行参数
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # LoRA隐含使用PEFT
    if model_args.use_lora:
        model_args.use_peft = True

    return model_args, data_args, training_args


def main():
    """主函数"""
    # 解析参数
    model_args, data_args, training_args = parse_args()

    # 确保使用CPU，禁用CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # 设置随机种子
    set_seed(training_args.seed)

    # 优化内存
    optimize_memory_usage()

    # 检查可用内存
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    logger.info(f"系统可用内存: {available_memory:.2f} GB")

    # 内存不足时自动调整量化级别
    if available_memory < 8 and model_args.quantization == "8bit":
        logger.warning(f"可用内存不足8GB，自动切换到4bit量化")
        model_args.quantization = "4bit"

    if available_memory < 5 and data_args.max_samples is None:
        max_samples = int(min(5000, 2000 + (available_memory * 1000)))
        logger.warning(f"可用内存有限，自动限制训练样本数为: {max_samples}")
        data_args.max_samples = max_samples

    # 处理输出目录
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 自动优化DeepSpeed配置
    if training_args.auto_optimize_ds_config:
        logger.info("自动优化DeepSpeed配置...")
        generate_optimized_ds_config(training_args.ds_config_path, overwrite=True)

    # 加载模型和分词器
    logger.info(f"加载模型: {model_args.model_name_or_path}")
    try:
        model, tokenizer = load_base_model(
            model_name_or_path=model_args.model_name_or_path,
            use_peft=model_args.use_peft,
            quantization=model_args.quantization,
            max_memory=model_args.max_memory
        )
    except ImportError as e:
        if "bitsandbytes" in str(e) or "accelerate" in str(e):
            logger.error("量化依赖缺失，尝试禁用量化并重新加载模型")
            model, tokenizer = load_base_model(
                model_name_or_path=model_args.model_name_or_path,
                use_peft=model_args.use_peft,
                quantization=None,  # 禁用量化
                max_memory=model_args.max_memory
            )
        else:
            raise

    # 如果使用LoRA，添加适配器
    if model_args.use_lora:
        logger.info("添加LoRA适配器...")
        peft_config = create_peft_config(
            lora_r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules
        )
        model = add_lora_to_model(model, peft_config)

    # 加载数据集
    logger.info("加载数据集...")
    dataset = load_training_dataset(
        dataset_name=data_args.dataset_name,
        data_path=data_args.data_path,
        data_files=data_args.data_files,
        text_column=data_args.text_column,
        max_samples=data_args.max_samples,
        seed=training_args.seed
    )

    # 准备数据集
    logger.info("准备数据集...")
    if data_args.instruction_format:
        # 使用指令格式
        tokenized_dataset = prepare_instruction_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            instruction_column=data_args.instruction_column,
            input_column=data_args.input_column,
            output_column=data_args.output_column,
            max_seq_length=data_args.max_seq_length
        )
    else:
        # 使用标准格式
        tokenized_dataset = prepare_dataset_for_training(
            dataset=dataset,
            tokenizer=tokenizer,
            text_column=data_args.text_column,
            max_seq_length=data_args.max_seq_length
        )

    # 创建训练参数 - 针对CPU进行特殊配置
    # 禁用FP16和BF16，因为在CPU上不适用
    training_args.fp16 = False
    training_args.bf16 = False

    trainer_args = create_training_args(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_ratio=training_args.warmup_ratio,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        fp16=False,  # 确保禁用FP16
        bf16=False,  # 确保禁用BF16
        deepspeed=training_args.ds_config_path if os.path.exists(training_args.ds_config_path) else None
    )

    # 创建训练器
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        args=trainer_args
    )

    # 确定恢复训练的检查点
    last_checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        last_checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # 训练模型
    logger.info("开始训练（CPU模式）...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 保存模型
    logger.info(f"保存模型到 {training_args.output_dir}")
    save_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        output_dir=training_args.output_dir,
        peft_only=model_args.use_peft
    )

    logger.info("训练完成！")


if __name__ == "__main__":
    main()