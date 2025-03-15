"""
ChatGLM-CPU-Trainer 量化工具模块
"""

import logging
import torch
import gc


def load_quantized_model(model_class, model_path, quantization="None", **kwargs):
    """
    加载量化模型的辅助函数

    参数:
        model_class: 模型类 (通常是AutoModel)
        model_path: 模型路径或名称
        quantization: 量化类型 ("4bit", "8bit", "None")
        **kwargs: 传递给模型加载函数的其他参数

    返回:
        加载的模型
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载模型: {model_path} (量化: {quantization})")

    # 清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 设置默认参数
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32
    }
    model_kwargs.update(kwargs)

    try:
        # 尝试执行量化
        if quantization == "4bit":
            logger.info("使用4bit量化...")
            # 确保导入必要的库
            try:
                import bitsandbytes as bnb

                # 记录版本信息以便调试
                logger.info(f"bitsandbytes版本: {bnb.__version__}")

                # 设置4bit量化配置
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_use_double_quant"] = True
                model_kwargs["bnb_4bit_quant_type"] = "nf4"
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float32
                model_kwargs["device_map"] = {"": "cpu"}
                if "torch_dtype" in model_kwargs:
                    del model_kwargs["torch_dtype"]

            except ImportError as e:
                logger.error(f"无法导入量化所需的库: {str(e)}")
                logger.warning("将使用非量化模式")
                quantization = "None"

        elif quantization == "8bit":
            logger.info("使用8bit量化...")
            try:
                import bitsandbytes as bnb

                # 记录版本信息以便调试
                logger.info(f"bitsandbytes版本: {bnb.__version__}")

                # 设置8bit量化配置
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = {"": "cpu"}
                if "torch_dtype" in model_kwargs:
                    del model_kwargs["torch_dtype"]

            except ImportError as e:
                logger.error(f"无法导入量化所需的库: {str(e)}")
                logger.warning("将使用非量化模式")
                quantization = "None"

        # 加载模型
        logger.info(f"正在加载模型，参数: {model_kwargs}")
        model = model_class.from_pretrained(model_path, **model_kwargs)
        logger.info("模型加载成功!")

        # 检查模型是否确实启用了量化
        if quantization != "None":
            logger.info("模型已加载并应用了量化设置")

        return model

    except Exception as e:
        logger.error(f"加载模型时发生错误: {str(e)}")

        # 如果量化失败，尝试非量化模式
        if quantization != "None":
            logger.warning("尝试使用非量化模式重新加载...")
            try:
                model = model_class.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                logger.info("使用非量化模式成功加载模型!")
                return model
            except Exception as e2:
                logger.error(f"非量化模式加载也失败: {str(e2)}")

        # 如果所有尝试都失败，重新抛出异常
        raise