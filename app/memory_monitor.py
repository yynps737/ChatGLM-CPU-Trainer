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
from pathlib import Path
import traceback
from collections import Counter

class MemoryMonitor:
    """内存使用监控类，用于追踪系统和进程的内存使用情况"""

    def __init__(self, logger=None, check_interval=30, warning_threshold=0.85,
                 max_objects_to_analyze=1000, log_dir="/app/data/output"):
        """
        初始化内存监控器

        参数:
            logger: 日志记录器，如果为None则创建一个新的
            check_interval: 检查内存间隔时间(秒)
            warning_threshold: 内存使用警告阈值(0-1)，超过该值将发出警告
            max_objects_to_analyze: 内存分析时检查的最大对象数
            log_dir: 日志文件目录
        """
        self.logger = logger or logging.getLogger(__name__)
        self.check_interval = check_interval
        self.warning_threshold = warning_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process(os.getpid())
        self.max_objects_to_analyze = max_objects_to_analyze

        # 记录最大内存使用量
        self.peak_system_memory = 0
        self.peak_process_memory = 0
        self.peak_timestamp = None

        # 动态阈值计算
        self.memory_history = []
        self.max_history_len = 10  # 保留最近10次测量结果

        # 创建内存日志文件
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        self.log_file = str(log_dir_path / "memory_usage.csv")

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,system_percent,process_rss_mb,process_vms_mb,available_memory_gb,warning_threshold\n")

    def get_memory_info(self):
        """
        获取当前内存使用信息

        返回:
            dict: 包含内存使用信息的字典
        """
        try:
            # 系统内存信息
            sys_memory = psutil.virtual_memory()

            # 进程内存信息
            proc_memory = self.process.memory_info()

            # 更新峰值
            if sys_memory.percent > self.peak_system_memory:
                self.peak_system_memory = sys_memory.percent
                self.peak_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            self.peak_process_memory = max(self.peak_process_memory, proc_memory.rss)

            # 记录内存历史
            if len(self.memory_history) >= self.max_history_len:
                self.memory_history.pop(0)
            self.memory_history.append(sys_memory.percent)

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
        except Exception as e:
            self.logger.error(f"获取内存信息出错: {e}")
            return {
                'system': {'total': 0, 'available': 0, 'used': 0, 'percent': 0},
                'process': {'rss': 0, 'vms': 0, 'percent': 0}
            }

    def log_memory_usage(self, memory_info=None):
        """
        将内存使用记录到CSV文件

        参数:
            memory_info: 可选的内存信息，如果为None则获取当前值
        """
        try:
            mem_info = memory_info or self.get_memory_info()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            available_gb = mem_info['system']['available'] / (1024**3)

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp},{mem_info['system']['percent']:.2f},"
                        f"{mem_info['process']['rss']/(1024**2):.2f},"
                        f"{mem_info['process']['vms']/(1024**2):.2f},"
                        f"{available_gb:.2f},"
                        f"{self._get_dynamic_threshold():.2f}\n")
        except Exception as e:
            self.logger.error(f"写入内存日志出错: {e}")

    def _get_dynamic_threshold(self):
        """
        基于历史内存使用计算动态警告阈值
        """
        if not self.memory_history:
            return self.warning_threshold * 100

        # 如果当前内存使用率增长迅速，降低阈值以提前预警
        if len(self.memory_history) >= 3:
            recent_trend = self.memory_history[-1] - self.memory_history[-3]
            if recent_trend > 10:  # 如果3个时间点内增加超过10%
                return max(self.warning_threshold * 95, self.memory_history[-1] + 5)

        return self.warning_threshold * 100

    def print_memory_info(self, detailed=False):
        """
        打印当前内存使用信息

        参数:
            detailed: 是否打印详细信息
        """
        try:
            mem_info = self.get_memory_info()

            # 基本信息
            self.logger.info(f"系统内存使用率: {mem_info['system']['percent']:.2f}%")
            self.logger.info(f"进程内存使用: {mem_info['process']['rss'] / (1024 ** 2):.2f}MB")

            # 记录到CSV
            self.log_memory_usage(mem_info)

            # 动态阈值
            dynamic_threshold = self._get_dynamic_threshold()

            # 检查是否接近内存限制
            if mem_info['system']['percent'] > dynamic_threshold:
                self.logger.warning(
                    f"警告: 系统内存使用率较高({mem_info['system']['percent']:.2f}%)，超过动态阈值({dynamic_threshold:.2f}%)!")

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
                self.logger.info(f"峰值系统内存使用率: {self.peak_system_memory:.2f}% (记录于 {self.peak_timestamp or '未知'})")

                self.logger.info(f"进程常驻内存(RSS): {proc_mem['rss'] / (1024 ** 2):.2f}MB")
                self.logger.info(f"进程虚拟内存(VMS): {proc_mem['vms'] / (1024 ** 2):.2f}MB")
                self.logger.info(f"峰值进程内存使用: {self.peak_process_memory / (1024 ** 2):.2f}MB")

                # 如果有PyTorch相关内存信息
                if torch.cuda.is_available():
                    self.logger.info(f"CUDA已分配内存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f}MB")
                    self.logger.info(f"CUDA缓存内存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f}MB")

                # 如果内存使用超过阈值，执行Python对象分析
                if mem_info['system']['percent'] > dynamic_threshold * 0.75:
                    self._analyze_python_objects()

        except Exception as e:
            self.logger.error(f"打印内存信息出错: {e}")
            self.logger.error(traceback.format_exc())

    def _analyze_python_objects(self):
        """分析Python对象内存使用情况, 使用采样方法提高效率"""
        try:
            import sys
            import random

            # 获取所有对象 - 潜在的高内存操作
            all_objects = gc.get_objects()
            total_objects = len(all_objects)
            self.logger.info(f"Python对象总数: {total_objects}")

            # 如果对象太多，采样分析
            if total_objects > self.max_objects_to_analyze:
                self.logger.info(f"对象数量过多，将对{self.max_objects_to_analyze}个样本进行分析")
                objects_to_analyze = random.sample(all_objects, self.max_objects_to_analyze)
            else:
                objects_to_analyze = all_objects

            # 使用Counter优化统计对象类型
            type_counts = Counter(type(obj).__name__ for obj in objects_to_analyze)

            # 分析大型对象
            large_torch_tensors = 0
            large_lists = 0
            large_dicts = 0
            large_numpy_arrays = 0

            for obj in objects_to_analyze:
                try:
                    if isinstance(obj, torch.Tensor) and obj.numel() > 1000000:
                        large_torch_tensors += 1
                    elif isinstance(obj, list) and len(obj) > 10000:
                        large_lists += 1
                    elif isinstance(obj, dict) and len(obj) > 10000:
                        large_dicts += 1
                    elif hasattr(obj, 'size') and hasattr(obj, 'dtype') and hasattr(obj, 'shape') and hasattr(obj, 'size') > 1000000:
                        # 可能是numpy数组
                        large_numpy_arrays += 1
                except:
                    pass

            # 打印前5个最常见对象类型
            top_types = type_counts.most_common(5)
            self.logger.info(f"最常见的对象类型: {top_types}")

            if large_torch_tensors > 0:
                self.logger.info(f"发现{large_torch_tensors}个大型Torch张量")
            if large_lists > 0:
                self.logger.info(f"发现{large_lists}个大型列表")
            if large_dicts > 0:
                self.logger.info(f"发现{large_dicts}个大型字典")
            if large_numpy_arrays > 0:
                self.logger.info(f"发现{large_numpy_arrays}个大型NumPy数组")

            # 主动清理临时变量
            del all_objects, objects_to_analyze, type_counts

        except Exception as e:
            self.logger.error(f"分析Python对象出错: {e}")

    def _try_free_memory(self):
        """尝试释放内存"""
        self.logger.info("尝试释放内存...")

        # 强制垃圾回收
        try:
            gc.collect()
            # 清除PyTorch缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("已执行垃圾回收")
        except Exception as e:
            self.logger.error(f"执行垃圾回收出错: {e}")

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
        fail_count = 0

        while self.monitoring:
            try:
                self.print_memory_info()
                fail_count = 0  # 重置失败计数
            except Exception as e:
                self.logger.error(f"内存监控遇到错误: {e}")
                fail_count += 1

                # 如果连续失败超过3次，休息一会
                if fail_count >= 3:
                    self.logger.warning("监控连续失败，暂停1分钟")
                    time.sleep(60)
                    fail_count = 0

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
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            try:
                self.monitor_thread.join(timeout=1.0)
            except:
                pass

        # 汇总内存使用情况
        self.logger.info("=" * 40)
        self.logger.info("内存监控汇总:")
        self.logger.info(f"峰值系统内存使用率: {self.peak_system_memory:.2f}% (记录于 {self.peak_timestamp or '未知'})")
        self.logger.info(f"峰值进程内存使用: {self.peak_process_memory / (1024 ** 2):.2f}MB")
        self.logger.info(f"内存使用日志已保存到: {self.log_file}")

        # 打印当前内存状态
        try:
            mem_info = self.get_memory_info()
            self.logger.info(f"最终系统内存使用率: {mem_info['system']['percent']:.2f}%")
            self.logger.info(f"最终进程内存使用: {mem_info['process']['rss'] / (1024 ** 2):.2f}MB")
        except:
            pass

        self.logger.info("=" * 40)
        self.logger.info("内存监控已停止")