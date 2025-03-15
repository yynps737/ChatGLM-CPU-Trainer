"""
ChatGLM-CPU-Trainer 内存监控模块

该模块提供了监控内存使用的工具，帮助用户在训练过程中
及时发现潜在的内存问题。
"""

import os
import logging
import psutil
import gc
import torch
from threading import Thread
import time
import json
from pathlib import Path


class MemoryMonitor:
    """内存使用监控类，用于追踪系统和进程的内存使用情况"""

    def __init__(self, logger=None, check_interval=30, warning_threshold=0.85):
        """
        初始化内存监控器

        参数:
            logger: 日志记录器，如果为None则创建一个新的
            check_interval: 检查内存间隔时间(秒)
            warning_threshold: 内存使用警告阈值(0-1)，超过该值将发出警告
        """
        self.logger = logger or logging.getLogger(__name__)
        self.check_interval = check_interval
        self.warning_threshold = warning_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process(os.getpid())

        # 记录最大内存使用量
        self.peak_system_memory = 0
        self.peak_process_memory = 0

        # 创建内存日志文件
        self.log_file = "/app/data/output/memory_usage.csv"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            f.write("timestamp,system_percent,process_rss_mb,process_vms_mb\n")

    def get_memory_info(self):
        """
        获取当前内存使用信息

        返回:
            dict: 包含内存使用信息的字典
        """
        # 系统内存信息
        sys_memory = psutil.virtual_memory()

        # 进程内存信息
        proc_memory = self.process.memory_info()

        # 更新峰值
        self.peak_system_memory = max(self.peak_system_memory, sys_memory.percent)
        self.peak_process_memory = max(self.peak_process_memory, proc_memory.rss)

        # 收集Python对象信息
        gc.collect()  # 尝试回收不再使用的对象

        return {
            'system': {
                'total': sys_memory.total,
                'available': sys_memory.available,
                'used': sys_memory.used,
                'percent': sys_memory.percent
            },
            'process': {
                'rss': proc_memory.rss,  # 常驻内存
                'vms': proc_memory.vms,  # 虚拟内存
                'percent': self.process.memory_percent()
            }
        }

    def print_memory_info(self, detailed=False):
        """
        打印当前内存使用信息

        参数:
            detailed: 是否打印详细信息
        """
        mem_info = self.get_memory_info()

        # 基本信息
        self.logger.info(f"系统内存使用率: {mem_info['system']['percent']}%")
        self.logger.info(f"进程内存使用: {mem_info['process']['rss'] / (1024 ** 2):.2f}MB")

        # 记录到CSV
        with open(self.log_file, 'a') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{mem_info['system']['percent']:.2f},{mem_info['process']['rss']/(1024**2):.2f},{mem_info['process']['vms']/(1024**2):.2f}\n")

        # 检查是否接近内存限制
        if mem_info['system']['percent'] > self.warning_threshold * 100:
            self.logger.warning(
                f"警告: 系统内存使用率较高({mem_info['system']['percent']}%)，可能影响性能或导致OOM错误!")

            # 提供优化建议
            self.logger.warning("建议: 考虑减少max_seq_length或max_samples参数，或切换到更低内存的配置。")

            # 主动触发内存回收
            self._try_free_memory()

        # 详细信息
        if detailed:
            sys_mem = mem_info['system']
            proc_mem = mem_info['process']

            self.logger.info(f"系统总内存: {sys_mem['total'] / (1024 ** 3):.2f}GB")
            self.logger.info(f"系统可用内存: {sys_mem['available'] / (1024 ** 3):.2f}GB")
            self.logger.info(f"系统已用内存: {sys_mem['used'] / (1024 ** 3):.2f}GB")
            self.logger.info(f"峰值系统内存使用率: {self.peak_system_memory:.2f}%")

            self.logger.info(f"进程常驻内存(RSS): {proc_mem['rss'] / (1024 ** 2):.2f}MB")
            self.logger.info(f"进程虚拟内存(VMS): {proc_mem['vms'] / (1024 ** 2):.2f}MB")
            self.logger.info(f"峰值进程内存使用: {self.peak_process_memory / (1024 ** 2):.2f}MB")

            # 如果有PyTorch相关内存信息
            if torch.cuda.is_available():
                self.logger.info(f"CUDA已分配内存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f}MB")
                self.logger.info(f"CUDA缓存内存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f}MB")

            # 获取当前Python内存使用情况
            import sys
            self.logger.info(f"Python对象数量: {len(gc.get_objects())}")
            # 找出最大的对象类型
            type_sizes = {}
            for obj in gc.get_objects():
                obj_type = str(type(obj))
                if obj_type not in type_sizes:
                    type_sizes[obj_type] = 0
                type_sizes[obj_type] += 1

            # 打印前5个最常见对象类型
            top_types = sorted(type_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info(f"最常见的对象类型: {top_types}")

    def _try_free_memory(self):
        """尝试释放内存"""
        self.logger.info("尝试释放内存...")
        # 强制垃圾回收
        gc.collect()
        # 清除PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 提示手动干预
        self.logger.info("建议: 如果内存使用持续增长，可能需要手动干预或减少训练参数。")

        # 输出部分设置建议
        mem_info = self.get_memory_info()
        sys_percent = mem_info['system']['percent']

        if sys_percent > 95:
            self.logger.warning("系统内存使用极高! 建议立即减少batch_size或停止训练")
        elif sys_percent > 85:
            self.logger.warning("建议将max_samples减半或将量化级别设为4bit")

    def _monitor_loop(self):
        """内存监控循环"""
        self.logger.info("内存监控循环已启动")
        while self.monitoring:
            try:
                self.print_memory_info()
            except Exception as e:
                self.logger.error(f"内存监控遇到错误: {e}")

            time.sleep(self.check_interval)

        self.logger.info("内存监控循环已终止")

    def start_monitoring(self):
        """启动内存监控"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"内存监控已启动 (检查间隔: {self.check_interval}秒)")

    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

        # 汇总内存使用情况
        self.logger.info("=" * 40)
        self.logger.info("内存监控汇总:")
        self.logger.info(f"峰值系统内存使用率: {self.peak_system_memory:.2f}%")
        self.logger.info(f"峰值进程内存使用: {self.peak_process_memory / (1024 ** 2):.2f}MB")
        self.logger.info(f"内存使用日志已保存到: {self.log_file}")
        self.logger.info("=" * 40)

        self.logger.info("内存监控已停止")

# 使用示例：
# monitor = MemoryMonitor()
# monitor.start_monitoring()
# ... 进行训练 ...
# monitor.stop_monitoring()