import os
import sys
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModel
)
from peft import (
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType
)

logger = logging.getLogger(__name__)


def is_chatglm_model(model_name_or_path):
    """判断是否为ChatGLM模型"""
    return "chatglm" in model_name_or_path.lower() or "glm" in model_name_or_path.lower()


def load_base_model(
        model_name_or_path: str,
        use_peft: bool = False,
        quantization: str = None,
        task_type: str = "causal_lm",
        max_memory=None
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    加载基础模型和分词器

    Args:
        model_name_or_path: 模型名称或路径
        use_peft: 是否使用PEFT/LoRA
        quantization: 量化类型 ("4bit", "8bit" 或 None)
        task_type: 任务类型 ("causal_lm" 或 "seq_cls")
        max_memory: 每个GPU的最大内存

    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    # 检查Python版本
    if not (sys.version_info.major == 3 and 8 <= sys.version_info.minor <= 13):
        logger.warning(
            f"当前Python版本 {sys.version_info.major}.{sys.version_info.minor} 可能不兼容，推荐使用Python 3.8-3.13")

    # 确定设备和低精度设置
    load_in_8bit = quantization == "8bit"
    load_in_4bit = quantization == "4bit"

    logger.info(f"加载模型: {model_name_or_path}")

    # 首先加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    # 基础模型加载参数
    model_kwargs = {
        "trust_remote_code": True
    }

    # 应用量化设置 - CPU环境特别处理
    if load_in_8bit or load_in_4bit:
        try:
            import bitsandbytes as bnb
            from packaging import version

            # 检查bitsandbytes版本
            required_version = "0.37.0"
            if version.parse(bnb.__version__) < version.parse(required_version):
                logger.warning(f"bitsandbytes版本 {bnb.__version__} 可能过低，推荐 >= {required_version}")

            if load_in_8bit:
                logger.info("使用8位量化加载模型")
                model_kwargs.update({"load_in_8bit": True})
            else:
                logger.info("使用4位量化加载模型")
                model_kwargs.update({"load_in_4bit": True})
        except ImportError:
            os_name = "Windows" if sys.platform.startswith("win") else "Linux"
            install_cmd = "pip install -U bitsandbytes-windows>=0.37.0" if os_name == "Windows" else "pip install -U bitsandbytes>=0.37.0"

            logger.warning("未找到bitsandbytes库或版本不兼容，改为使用浮点精度加载")
            logger.warning(f"提示: 运行 '{install_cmd}' 安装所需库")
            load_in_8bit = False
            load_in_4bit = False
            model_kwargs.update({"torch_dtype": torch.float32})
    else:
        logger.info("使用原始精度加载模型")
        model_kwargs.update({"torch_dtype": torch.float32})  # 使用float32代替bfloat16，更适合CPU

    # 设置为CPU设备，确保不使用GPU
    # 注释掉device_map="auto"，因为在纯CPU环境下，这可能会导致问题
    model_kwargs["device_map"] = None  # 使用None代替"auto"，显式指定不使用device mapping

    # 根据模型类型和任务类型加载合适的模型
    is_glm = is_chatglm_model(model_name_or_path)

    # 捕获模型加载的错误并提供友好提示
    try:
        if is_glm:
            logger.info("检测到ChatGLM模型，使用专用加载方式")
            model = AutoModel.from_pretrained(
                model_name_or_path,
                **model_kwargs
            )
        elif task_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs
            )
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
    except ImportError as e:
        # 处理可能的依赖问题
        if "bitsandbytes" in str(e) or "accelerate" in str(e):
            os_name = "Windows" if sys.platform.startswith("win") else "Linux"
            bnb_install = "pip install -U bitsandbytes-windows>=0.37.0" if os_name == "Windows" else "pip install -U bitsandbytes>=0.37.0"

            logger.error("缺少8位/4位量化所需的依赖，尝试安装:")
            logger.error(f"  pip install -U accelerate")
            logger.error(f"  {bnb_install}")
            logger.error("或重新运行时使用 --quantization None 禁用量化")
            raise ImportError("量化依赖缺失，详见上述日志") from e
        else:
            # 其他导入错误，直接抛出
            raise

    # 确保模型在CPU上
    model = model.to("cpu")

    # 确保分词器有正确的特殊标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_peft_config(
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=None,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        bias="none"
) -> LoraConfig:
    """
    创建LoRA配置

    Args:
        lora_r: LoRA注意力维度
        lora_alpha: LoRA缩放参数
        lora_dropout: LoRA的Dropout率
        target_modules: 要应用LoRA的目标模块
        task_type: 任务类型
        inference_mode: 是否为推理模式
        bias: 偏置参数训练方式

    Returns:
        LoRA配置对象
    """
    # 为ChatGLM模型定制的目标模块
    if target_modules is None:
        target_modules = [
            "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
        ]

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        inference_mode=inference_mode
    )


def add_lora_to_model(model, peft_config) -> PeftModel:
    """
    将LoRA适配器添加到模型

    Args:
        model: 要添加LoRA的基础模型
        peft_config: LoRA配置

    Returns:
        添加了LoRA的PEFT模型
    """
    logger.info("向模型添加PEFT/LoRA适配器")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def save_model_and_tokenizer(model, tokenizer, output_dir, peft_only=False):
    """
    保存模型和分词器

    Args:
        model: 要保存的模型
        tokenizer: 要保存的分词器
        output_dir: 输出目录
        peft_only: 是否只保存PEFT权重
    """
    # 标准化输出路径
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if peft_only and isinstance(model, PeftModel):
        logger.info(f"仅保存PEFT权重到 {output_dir}")
        model.save_pretrained(output_dir)
    else:
        logger.info(f"保存完整模型到 {output_dir}")
        model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)
    logger.info(f"已将模型和分词器保存到 {output_dir}")


def load_peft_model(base_model_path, adapter_path, quantization=None):
    """
    加载基础模型和PEFT适配器

    Args:
        base_model_path: 基础模型路径
        adapter_path: 适配器路径
        quantization: 量化类型

    Returns:
        model: 加载的PEFT模型
        tokenizer: 加载的分词器
    """
    # 标准化路径
    base_model_path = os.path.normpath(base_model_path)
    adapter_path = os.path.normpath(adapter_path)

    logger.info(f"加载基础模型 {base_model_path}")
    model, tokenizer = load_base_model(base_model_path, quantization=quantization)

    logger.info(f"从 {adapter_path} 加载PEFT适配器")
    model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer