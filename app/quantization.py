"""
ChatGLM-CPU-Trainer 量化工具模块

该模块提供专门的模型量化功能，通过降低模型的精度（位宽）来减少内存消耗，
使大型语言模型能够在低内存环境中运行。目前支持4-bit和8-bit量化。
"""

import logging
import torch
import gc
import os
import traceback
from packaging import version

def check_bitsandbytes_compatibility():
    """检查bitsandbytes版本兼容性"""
    logger = logging.getLogger(__name__)
    try:
        import bitsandbytes as bnb
        import transformers

        bnb_version = bnb.__version__
        tf_version = transformers.__version__

        logger.info(f"bitsandbytes版本: {bnb_version}")
        logger.info(f"transformers版本: {tf_version}")

        if version.parse(bnb_version) < version.parse("0.37.0"):
            logger.warning(f"bitsandbytes版本过低 ({bnb_version})，推荐 0.37.0 或更高版本")
            return False

        if version.parse(tf_version) < version.parse("4.30.0"):
            logger.warning(f"transformers版本过低 ({tf_version})，可能不完全支持量化功能，推荐 4.30.0 或更高版本")
            return False

        return True
    except ImportError as e:
        logger.error(f"检查bitsandbytes兼容性时出错: {e}")
        return False
    except Exception as e:
        logger.error(f"验证版本时出错: {e}")
        return False

def setup_4bit_config(logger):
    """设置4bit量化配置

    返回:
        dict: 量化配置参数
    """
    try:
        import bitsandbytes as bnb
        import transformers
        from packaging import version

        # 记录版本信息以便调试
        logger.info(f"bitsandbytes版本: {bnb.__version__}")
        logger.info(f"transformers版本: {transformers.__version__}")

        # 检查transformers版本是否支持4bit量化
        if version.parse(transformers.__version__) < version.parse("4.30.0"):
            logger.warning(f"当前transformers版本 ({transformers.__version__}) 可能不完全支持4bit量化，推荐4.30.0或更高版本")

        # 设置环境变量以优化CPU上的量化过程
        os.environ["BITSANDBYTES_CPU_RAM_PERCENT_USAGE"] = os.environ.get("BITSANDBYTES_CPU_RAM_PERCENT_USAGE", "95")

        # 设置4bit量化配置
        model_kwargs = {
            "load_in_4bit": True,  # 启用4bit量化加载
            "bnb_4bit_use_double_quant": True,  # 启用嵌套量化，进一步减少内存占用
            "bnb_4bit_quant_type": "nf4",  # 使用NF4（normalized float 4）量化类型
            "bnb_4bit_compute_dtype": torch.float32,  # 计算时使用的数据类型
            "device_map": {"": "cpu"},  # 将模型映射到CPU上
            "trust_remote_code": True  # 允许执行模型仓库中的自定义代码
        }

        logger.info("4bit量化配置设置完成")
        return model_kwargs

    except ImportError as e:
        logger.error(f"无法导入量化所需的库: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"设置4bit量化配置时出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def setup_8bit_config(logger):
    """设置8bit量化配置

    返回:
        dict: 量化配置参数
    """
    try:
        import bitsandbytes as bnb
        import transformers
        from packaging import version

        # 记录版本信息以便调试
        logger.info(f"bitsandbytes版本: {bnb.__version__}")
        logger.info(f"transformers版本: {transformers.__version__}")

        # 检查transformers版本是否支持8bit量化
        if version.parse(transformers.__version__) < version.parse("4.26.0"):
            logger.warning(f"当前transformers版本 ({transformers.__version__}) 可能不完全支持8bit量化，推荐4.26.0或更高版本")

        # 设置环境变量以优化CPU上的量化过程
        os.environ["BITSANDBYTES_CPU_RAM_PERCENT_USAGE"] = os.environ.get("BITSANDBYTES_CPU_RAM_PERCENT_USAGE", "95")

        # 设置8bit量化配置
        model_kwargs = {
            "load_in_8bit": True,  # 启用8bit量化加载
            "device_map": {"": "cpu"},  # 将模型映射到CPU上
            "trust_remote_code": True  # 允许执行模型仓库中的自定义代码
        }

        logger.info("8bit量化配置设置完成")
        return model_kwargs

    except ImportError as e:
        logger.error(f"无法导入量化所需的库: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"设置8bit量化配置时出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def verify_quantization(model, quantization, logger):
    """验证模型是否正确量化

    参数:
        model: 已加载的模型
        quantization: 量化类型
        logger: 日志记录器

    返回:
        bool: 是否成功量化
    """
    if quantization == "None":
        return True

    try:
        # 尝试查找量化的线性层
        is_quantized = False
        quantized_layers = 0
        total_linear_layers = 0

        for name, module in model.named_modules():
            if "Linear" in str(type(module)):
                total_linear_layers += 1
                if "Linear4bit" in str(type(module)) or "Linear8bit" in str(type(module)):
                    is_quantized = True
                    quantized_layers += 1
                    if quantized_layers <= 3:  # 只记录前几个量化层
                        logger.info(f"检测到量化层: {name} - {type(module).__name__}")

        if total_linear_layers > 0:
            quantization_ratio = quantized_layers / total_linear_layers
            logger.info(f"量化率: {quantization_ratio:.2%} ({quantized_layers}/{total_linear_layers})")

            if quantization_ratio < 0.5:
                logger.warning(f"量化率低于50%，模型可能未完全量化")
                if quantization_ratio > 0.1:
                    logger.info("部分层已量化，将继续使用")
                    return True
                else:
                    logger.warning("几乎没有层被量化，模型量化可能失败")
                    return False

        if not is_quantized:
            logger.warning(f"未检测到量化层，模型可能未正确量化为{quantization}")
            return False
        return True
    except Exception as e:
        logger.error(f"验证量化时出错: {e}")
        logger.error(traceback.format_exc())
        return False

def load_model_fallback(model_class, model_path, logger):
    """在量化失败时的模型加载回退方案

    参数:
        model_class: 模型类
        model_path: 模型路径
        logger: 日志记录器

    返回:
        加载的模型或引发异常
    """
    try:
        # 使用最基本的配置加载模型
        logger.info("尝试使用非量化模式加载模型...")

        # 清理内存先
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = model_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        logger.info("使用非量化模式成功加载模型!")
        logger.warning("注意: 非量化模式会占用更多内存，可能导致内存不足错误")
        return model
    except Exception as e:
        logger.error(f"非量化模式加载也失败: {e}")
        logger.error(traceback.format_exc())

        # 尝试使用低精度加载
        try:
            logger.info("尝试使用半精度(float16)加载模型...")
            model = model_class.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            logger.info("使用半精度成功加载模型!")
            return model
        except Exception as e2:
            logger.error(f"半精度加载也失败: {e2}")
            logger.error(traceback.format_exc())

        raise RuntimeError("无法加载模型，所有尝试均失败") from e

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

    量化说明:
        - None: 不进行量化，使用原始精度（通常是float32）
        - 8bit: 将模型参数量化为8位整数，内存占用约为原来的1/4
        - 4bit: 将模型参数量化为4位整数，内存占用约为原来的1/8
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载模型: {model_path} (量化: {quantization})")

    # 清理内存，确保有足够空间加载模型
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 设置默认参数
    model_kwargs = {
        "trust_remote_code": True,  # 必须为ChatGLM模型启用
        "torch_dtype": torch.float32  # 默认使用float32精度
    }
    model_kwargs.update(kwargs)

    # 根据量化类型设置配置
    try:
        if quantization == "4bit":
            # 确认环境兼容性
            if not check_bitsandbytes_compatibility():
                logger.warning("量化环境兼容性检查失败，将尝试继续但可能会出错")

            # 设置4bit量化配置
            try:
                quant_config = setup_4bit_config(logger)
                model_kwargs.update(quant_config)
                # 移除torch_dtype以避免冲突
                if "torch_dtype" in model_kwargs:
                    del model_kwargs["torch_dtype"]
            except Exception as e:
                logger.error(f"设置4bit量化配置失败: {e}")
                logger.warning("将使用非量化模式")
                quantization = "None"

        elif quantization == "8bit":
            # 确认环境兼容性
            if not check_bitsandbytes_compatibility():
                logger.warning("量化环境兼容性检查失败，将尝试继续但可能会出错")

            # 设置8bit量化配置
            try:
                quant_config = setup_8bit_config(logger)
                model_kwargs.update(quant_config)
                # 移除torch_dtype以避免冲突
                if "torch_dtype" in model_kwargs:
                    del model_kwargs["torch_dtype"]
            except Exception as e:
                logger.error(f"设置8bit量化配置失败: {e}")
                logger.warning("将使用非量化模式")
                quantization = "None"

        # 加载模型
        logger.info(f"正在加载模型，参数: {model_kwargs}")

        try:
            model = model_class.from_pretrained(model_path, **model_kwargs)
            logger.info("模型加载成功!")

            # 验证量化是否成功
            if quantization != "None":
                if verify_quantization(model, quantization, logger):
                    if quantization == "4bit":
                        logger.info("4-bit量化: 内存占用约为原始模型的1/8，性能略有影响")
                    elif quantization == "8bit":
                        logger.info("8-bit量化: 内存占用约为原始模型的1/4，性能影响很小")
                else:
                    logger.warning("量化验证失败，但模型已加载")
                    logger.info("将继续使用已加载的模型，但可能未达到预期的内存优化效果")

            # 输出模型结构信息
            try:
                param_count = sum(p.numel() for p in model.parameters())
                trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"模型总参数量: {param_count:,}")
                logger.info(f"可训练参数量: {trainable_param_count:,}")
                logger.info(f"参数可训练比例: {trainable_param_count/param_count:.2%}")
            except Exception as e:
                logger.warning(f"计算参数量出错: {e}")

            return model

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            logger.error(traceback.format_exc())

            # 如果量化失败，尝试非量化模式
            if quantization != "None":
                logger.warning("尝试使用非量化模式重新加载...")
                return load_model_fallback(model_class, model_path, logger)
            raise

    except Exception as e:
        logger.error(f"加载模型过程中发生错误: {e}")
        logger.error(traceback.format_exc())

        # 如果量化失败，尝试非量化模式
        if quantization != "None":
            logger.warning("尝试使用非量化模式重新加载...")
            return load_model_fallback(model_class, model_path, logger)
        raise