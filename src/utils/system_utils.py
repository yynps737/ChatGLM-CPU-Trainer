import os
import json
import platform
import subprocess
import torch
import logging
import psutil
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, any]:
    """获取系统信息"""
    cpu_info = {}

    try:
        # 获取CPU型号
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_info["model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
        elif platform.system() == "Darwin":  # macOS
            cpu_info["model"] = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info["model"] = line.split(":")[1].strip()
                        break
    except:
        cpu_info["model"] = platform.processor()

    info = {
        "platform": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "processor": cpu_info.get("model", platform.processor()),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "has_cuda": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    # 识别AMD处理器
    is_amd = False
    if info["processor"] and "AMD" in info["processor"]:
        is_amd = True
        info["is_amd"] = True

    return info


def optimize_memory_usage() -> bool:
    """优化内存使用"""
    # 清除任何可能的CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 垃圾回收
    logger.info("进行垃圾回收...")
    import gc
    gc.collect()

    # 获取当前内存状态
    vm = psutil.virtual_memory()
    logger.info(f"内存使用率: {vm.percent}%")
    logger.info(f"可用内存: {vm.available / (1024 ** 3):.2f} GB")

    # 优化CPU线程设置
    system_info = get_system_info()
    logical_cores = system_info["cpu_count_logical"]

    # 如果检测到AMD Ryzen，提供特定优化
    if system_info.get("is_amd", False):
        logger.info("检测到AMD Ryzen处理器，应用特定优化...")
        # 环境变量已在Docker中设置，这里只是提示
        logger.info(f"建议设置 OMP_NUM_THREADS={logical_cores} 和 MKL_NUM_THREADS={logical_cores}")

    return True


def generate_optimized_ds_config(
        output_path: str = "configs/default_config.json",
        base_config_path: Optional[str] = None,
        overwrite: bool = False
) -> bool:
    """生成优化的DeepSpeed配置

    Args:
        output_path: 输出路径
        base_config_path: 基本配置路径
        overwrite: 是否覆盖现有文件

    Returns:
        bool: 是否成功生成配置
    """
    # 如果文件已存在且不覆盖，直接返回
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"DeepSpeed配置文件 {output_path} 已存在，不覆盖")
        return True

    # 获取系统信息
    system_info = get_system_info()

    # 如果提供了基本配置，加载它
    if base_config_path and os.path.exists(base_config_path):
        logger.info(f"基于 {base_config_path} 生成优化配置")
        with open(base_config_path, 'r') as f:
            config = json.load(f)
    else:
        logger.info("创建新的DeepSpeed CPU优化配置")
        # 创建CPU优化配置
        config = {
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5000000,
                "stage3_prefetch_bucket_size": 5000000,
                "stage3_param_persistence_threshold": 100000
            },
            "gradient_accumulation_steps": 32,
            "gradient_clipping": 1.0,
            "steps_per_print": 10,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }

    # CPU优化：使用ZeRO-3和更高的梯度累积
    memory_gb = system_info["memory_available_gb"]
    logical_cores = system_info["cpu_count_logical"]

    # 针对AMD Ryzen 7950X的特定优化
    if "AMD Ryzen" in system_info["processor"]:
        logger.info("检测到AMD Ryzen处理器，应用特定优化...")
        if memory_gb >= 12:
            config["train_micro_batch_size_per_gpu"] = 1
            config["gradient_accumulation_steps"] = 32
        else:
            config["train_micro_batch_size_per_gpu"] = 1
            config["gradient_accumulation_steps"] = 64

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 写入配置文件
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"已生成CPU优化的DeepSpeed配置: {output_path}")
    return True


def auto_detect_batch_size(
        model,
        tokenizer,
        start_batch_size=8,
        max_length=512,
        min_batch_size=1
) -> int:
    """自动检测最大批大小

    Args:
        model: 模型
        tokenizer: 分词器
        start_batch_size: 起始批大小
        max_length: 最大序列长度
        min_batch_size: 最小批大小

    Returns:
        int: 最大可用批大小
    """
    logger.info(f"开始自动检测批大小 (起始值: {start_batch_size})...")

    # 将模型移至评估模式
    model.eval()

    # 从start_batch_size开始递减测试
    batch_size = start_batch_size
    while batch_size >= min_batch_size:
        try:
            # 创建示例输入
            sample_input = tokenizer.pad({
                "input_ids": [[1] * max_length] * batch_size
            }, return_tensors="pt")

            # 尝试前向传播
            with torch.no_grad():
                outputs = model(**sample_input)

            # 如果成功，退出循环
            logger.info(f"找到可用批大小: {batch_size}")
            return batch_size

        except RuntimeError as e:
            if "DefaultCPUAllocator: can't allocate memory" in str(e):
                logger.info(f"批大小 {batch_size} 导致内存溢出，尝试更小的值...")
                batch_size = batch_size // 2
            else:
                # 如果是其他错误，记录并返回最小值
                logger.error(f"检测批大小时出现错误: {e}")
                return min_batch_size

    # 如果所有批大小都失败，返回最小值
    logger.warning(f"所有批大小都失败，使用最小值 {min_batch_size}")
    return min_batch_size