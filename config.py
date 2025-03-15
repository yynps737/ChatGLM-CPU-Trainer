#!/usr/bin/env python
"""
ChatGLM-CPU-Trainer 配置生成工具

该脚本用于检测系统配置并生成适合的.env文件，取代原来的批处理脚本。
"""

import os
import sys
import platform
import psutil
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def detect_system_memory():
    """检测系统内存"""
    logger.info("检测系统内存...")

    try:
        # 获取系统内存
        system_memory = psutil.virtual_memory()
        total_memory_gb = system_memory.total / (1024 ** 3)
        logger.info(f"检测到系统内存: {total_memory_gb:.2f}GB")
        return total_memory_gb
    except Exception as e:
        logger.error(f"无法检测系统内存: {e}")
        logger.info("使用默认8GB配置")
        return 8


def get_memory_config(total_memory_gb):
    """根据内存大小确定配置"""
    if total_memory_gb < 6:
        return "4gb"
    elif total_memory_gb < 12:
        return "8gb"
    elif total_memory_gb < 24:
        return "16gb"
    else:
        return "32gb"


def generate_env_file(memory_config):
    """生成.env配置文件"""
    logger.info(f"创建.env配置文件，使用{memory_config}配置...")

    # 环境变量配置
    config = {
        "4gb": {
            "MEMORY_LIMIT": "3.8G",
            "NUM_THREADS": "2",
            "QUANT_LEVEL": "4bit",
            "MAX_SEQ_LEN": "32",
            "MAX_SAMPLES": "30",
            "LORA_R": "4",
            "BATCH_SIZE": "1",
            "GRAD_ACCUM": "32",
            "MAX_LENGTH": "512",
            "MONITOR_MEMORY": "true",
            "MEMORY_CHECK_INTERVAL": "30",
            "PERFORMANCE_LOG_STEPS": "50"
        },
        "8gb": {
            "MEMORY_LIMIT": "7.5G",
            "NUM_THREADS": "4",
            "QUANT_LEVEL": "4bit",
            "MAX_SEQ_LEN": "64",
            "MAX_SAMPLES": "200",
            "LORA_R": "8",
            "BATCH_SIZE": "1",
            "GRAD_ACCUM": "16",
            "MAX_LENGTH": "1024",
            "MONITOR_MEMORY": "true",
            "MEMORY_CHECK_INTERVAL": "60",
            "PERFORMANCE_LOG_STEPS": "100"
        },
        "16gb": {
            "MEMORY_LIMIT": "15G",
            "NUM_THREADS": "8",
            "QUANT_LEVEL": "8bit",
            "MAX_SEQ_LEN": "128",
            "MAX_SAMPLES": "800",
            "LORA_R": "16",
            "BATCH_SIZE": "2",
            "GRAD_ACCUM": "8",
            "MAX_LENGTH": "2048",
            "MONITOR_MEMORY": "true",
            "MEMORY_CHECK_INTERVAL": "120",
            "PERFORMANCE_LOG_STEPS": "200"
        },
        "32gb": {
            "MEMORY_LIMIT": "30G",
            "NUM_THREADS": "16",
            "QUANT_LEVEL": "None",
            "MAX_SEQ_LEN": "256",
            "MAX_SAMPLES": "2000",
            "LORA_R": "32",
            "BATCH_SIZE": "4",
            "GRAD_ACCUM": "4",
            "MAX_LENGTH": "4096",
            "MONITOR_MEMORY": "true",
            "MEMORY_CHECK_INTERVAL": "180",
            "PERFORMANCE_LOG_STEPS": "300"
        }
    }

    # 创建.env文件
    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"# 配置文件由config.py自动生成\n")
        f.write(f"MEMORY_CONFIG={memory_config}\n")

        # 写入选定配置的所有参数
        for key, value in config[memory_config].items():
            f.write(f"{key}={value}\n")

        # 添加通用配置
        f.write(f"HF_ENDPOINT=https://hf-mirror.com\n")

    logger.info(f".env文件已创建，使用{memory_config}配置")


def create_directories():
    """创建必要的目录结构"""
    logger.info("创建目录结构...")

    # 创建目录
    directories = [
        "data/input",
        "data/output",
        "models"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    logger.info("目录结构已创建")


def check_example_dataset():
    """检查示例数据集是否存在"""
    dataset_path = "data/input/dataset.txt"

    if not os.path.exists(dataset_path):
        logger.warning(f"未找到示例数据集文件 {dataset_path}")
        logger.info("提示: 您需要在训练前准备自己的数据集")
    else:
        logger.info(f"已找到示例数据集文件: {dataset_path}")


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("ChatGLM-CPU-Trainer 配置生成工具")
    logger.info("=" * 50)

    # 检测系统
    system = platform.system()
    logger.info(f"操作系统: {system}")

    # 检测内存
    total_memory_gb = detect_system_memory()

    # 确定内存配置
    memory_config = get_memory_config(total_memory_gb)
    logger.info(f"将使用{memory_config}内存优化配置")

    # 生成配置文件
    generate_env_file(memory_config)

    # 创建目录结构
    create_directories()

    # 检查示例数据集
    check_example_dataset()

    # 显示下一步
    logger.info("=" * 50)
    logger.info("配置完成！接下来请:")
    logger.info("1. 将训练数据放入data/input/dataset.txt文件")
    logger.info("2. 使用以下命令开始训练:")
    logger.info("   docker-compose run app train")
    logger.info("3. 使用以下命令测试模型:")
    logger.info("   docker-compose run app predict --prompt \"您的提示词\"")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()