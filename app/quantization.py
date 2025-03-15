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

        if bnb_version < "0.37.0":
            logger.warning(f"bitsandbytes版本过低 ({bnb_version})，推荐 0.37.0 或更高版本")
            return False

        if tf_version < "4.30.0":
            logger.warning(f"transformers版本过低 ({tf_version})，可能不完全支持量化功能，推荐 4.30.0 或更高版本")
            return False

        return True
    except ImportError as e:
        logger.error(f"检查bitsandbytes兼容性时出错: {e}")
        return False
    except Exception as e:
        logger.error(f"验证版本时出错: {e}")
        return False

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
        "trust_remote_code": True,  # 必须为ChatGLM模型启用，允许执行模型仓库中的自定义代码
        "torch_dtype": torch.float32  # 默认使用float32精度
    }
    model_kwargs.update(kwargs)

    # 如果需要使用量化，首先检查环境兼容性
    if quantization in ["4bit", "8bit"]:
        compat_ok = check_bitsandbytes_compatibility()
        if not compat_ok:
            logger.warning("量化环境兼容性检查失败，将尝试继续但可能会出错")

    try:
        # 尝试执行量化
        if quantization == "4bit":
            logger.info("使用4bit量化...")
            # 确保导入必要的库
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
                model_kwargs["load_in_4bit"] = True  # 启用4bit量化加载
                model_kwargs["bnb_4bit_use_double_quant"] = True  # 启用嵌套量化，进一步减少内存占用
                model_kwargs["bnb_4bit_quant_type"] = "nf4"  # 使用NF4（normalized float 4）量化类型，在保持精度的同时降低内存占用
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float32  # 计算时使用的数据类型
                model_kwargs["device_map"] = {"": "cpu"}  # 将模型映射到CPU上，重要！

                # 移除默认dtype，避免与量化配置冲突
                if "torch_dtype" in model_kwargs:
                    del model_kwargs["torch_dtype"]

                logger.info("4bit量化配置设置完成")

            except ImportError as e:
                logger.error(f"无法导入量化所需的库: {str(e)}")
                logger.warning("将使用非量化模式")
                quantization = "None"

            except Exception as e:
                logger.error(f"设置4bit量化配置时出错: {str(e)}")
                logger.error(traceback.format_exc())
                logger.warning("将使用非量化模式")
                quantization = "None"

        elif quantization == "8bit":
            logger.info("使用8bit量化...")
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
                model_kwargs["load_in_8bit"] = True  # 启用8bit量化加载
                model_kwargs["device_map"] = {"": "cpu"}  # 将模型映射到CPU上，重要！

                # 移除默认dtype，避免与量化配置冲突
                if "torch_dtype" in model_kwargs:
                    del model_kwargs["torch_dtype"]

                logger.info("8bit量化配置设置完成")

            except ImportError as e:
                logger.error(f"无法导入量化所需的库: {str(e)}")
                logger.warning("将使用非量化模式")
                quantization = "None"

            except Exception as e:
                logger.error(f"设置8bit量化配置时出错: {str(e)}")
                logger.error(traceback.format_exc())
                logger.warning("将使用非量化模式")
                quantization = "None"

        # 加载模型
        logger.info(f"正在加载模型，参数: {model_kwargs}")

        try:
            model = model_class.from_pretrained(model_path, **model_kwargs)
            logger.info("模型加载成功!")

            # 检查模型是否确实启用了量化
            if quantization != "None":
                logger.info("模型已加载并应用了量化设置")

                # 量化性能说明
                if quantization == "4bit":
                    logger.info("4-bit量化: 内存占用约为原始模型的1/8，性能略有影响")
                elif quantization == "8bit":
                    logger.info("8-bit量化: 内存占用约为原始模型的1/4，性能影响很小")

                # 验证模型是否真的量化了
                is_quantized = False
                try:
                    # 尝试查找量化的线性层
                    for name, module in model.named_modules():
                        if "Linear4bit" in str(type(module)) or "Linear8bit" in str(type(module)):
                            is_quantized = True
                            logger.info(f"检测到量化层: {name} - {type(module).__name__}")
                            break

                    if not is_quantized:
                        logger.warning(f"未检测到量化层，模型可能未正确量化为{quantization}")
                except:
                    # 忽略检测错误
                    pass
        except Exception as e:
            logger.error(f"加载模型时发生错误: {str(e)}")
            logger.error(traceback.format_exc())

            # 如果量化失败，尝试非量化模式
            if quantization != "None":
                logger.warning("尝试使用非量化模式重新加载...")
                try:
                    # 使用基本参数加载模型
                    model = model_class.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float32
                    )
                    logger.info("使用非量化模式成功加载模型!")
                    logger.warning("注意: 非量化模式会占用更多内存，可能导致内存不足错误")
                    return model
                except Exception as e2:
                    logger.error(f"非量化模式加载也失败: {str(e2)}")
                    logger.error(traceback.format_exc())
                    raise RuntimeError("无法加载模型，所有尝试均失败") from e2

            # 如果不是量化错误或非量化加载也失败，重新抛出异常
            raise

        return model

    except Exception as e:
        logger.error(f"加载模型时发生错误: {str(e)}")
        logger.error(traceback.format_exc())

        # 如果量化失败，尝试非量化模式
        if quantization != "None":
            logger.warning("尝试使用非量化模式重新加载...")
            try:
                # 使用基本参数加载模型
                model = model_class.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                logger.info("使用非量化模式成功加载模型!")
                logger.warning("注意: 非量化模式会占用更多内存，可能导致内存不足错误")
                return model
            except Exception as e2:
                logger.error(f"非量化模式加载也失败: {str(e2)}")
                logger.error(traceback.format_exc())

        # 如果所有尝试都失败，重新抛出异常
        raise RuntimeError("无法加载模型，所有尝试均失败") from e

# 量化参数说明：
# 1. load_in_4bit/load_in_8bit: 启用对应位宽的量化
# 2. bnb_4bit_use_double_quant: 对权重应用两级量化，进一步降低内存需求
# 3. bnb_4bit_quant_type: 量化类型，nf4保留更多数值信息
# 4. bnb_4bit_compute_dtype: 计算时使用的精度
# 5. device_map: 指定模型各部分加载到哪个设备，这里使用CPU