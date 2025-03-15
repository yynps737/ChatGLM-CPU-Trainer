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

        # 检查是否接近内存限制
        if mem_info['system']['percent'] > self.warning_threshold * 100:
            self.logger.warning(
                f"警告: 系统内存使用率较高({mem_info['system']['percent']}%)，可能影响性能或导致OOM错误!")

            # 提供优化建议
            self.logger.warning("建议: 考虑减少max_seq_length或max_samples参数，或切换到更低内存的配置。")

        # 详细信息
        if detailed:
            sys_mem = mem_info['system']
            proc_mem = mem_info['process']

            self.logger.info(f"系统总内存: {sys_mem['total'] / (1024 ** 3):.2f}GB")
            self.logger.info(f"系统可用内存: {sys_mem['available'] / (1024 ** 3):.2f}GB")
            self.logger.info(f"系统已用内存: {sys_mem['used'] / (1024 ** 3):.2f}GB")

            self.logger.info(f"进程常驻内存(RSS): {proc_mem['rss'] / (1024 ** 2):.2f}MB")
            self.logger.info(f"进程虚拟内存(VMS): {proc_mem['vms'] / (1024 ** 2):.2f}MB")

            # 如果有PyTorch相关内存信息
            if torch.cuda.is_available():
                self.logger.info(f"CUDA已分配内存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f}MB")
                self.logger.info(f"CUDA缓存内存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f}MB")

    def _monitor_loop(self):
        """内存监控循环"""
        while self.monitoring:
            try:
                self.print_memory_info()
            except Exception as e:
                self.logger.error(f"内存监控遇到错误: {e}")

            time.sleep(self.check_interval)

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
        self.logger.info("内存监控已停止")

# 使用示例：
# monitor = MemoryMonitor()
# monitor.start_monitoring()
# ... 进行训练 ...
# monitor.stop_monitoring()`