"""
内存监控脚本 - 在训练过程中监控内存使用情况
使用方法: python memory_monitor.py
"""

import os
import time
import psutil
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import numpy as np

# 配置
UPDATE_INTERVAL = 5  # 更新间隔（秒）
MAX_POINTS = 100  # 最大数据点数量
ALERT_THRESHOLD = 90  # 内存使用率警告阈值


class MemoryMonitor:
    def __init__(self):
        # 初始化数据存储
        self.timestamps = []
        self.memory_percent = []
        self.memory_used = []
        self.memory_available = []

        # 获取总内存
        self.total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB

        # 设置图表
        plt.style.use('ggplot')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('ChatGLM训练内存监控')

        # 配置子图1 - 内存百分比
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2)
        self.ax1.set_ylim(0, 100)
        self.ax1.set_ylabel('内存使用率 (%)')
        self.ax1.set_title(f'系统内存监控 (总内存: {self.total_memory:.2f} GB)')
        self.ax1.grid(True)

        # 警告线
        self.ax1.axhline(y=ALERT_THRESHOLD, color='r', linestyle='--', alpha=0.7)
        self.ax1.text(0.02, ALERT_THRESHOLD + 2, f'警告阈值 ({ALERT_THRESHOLD}%)',
                      color='r', transform=self.ax1.get_yaxis_transform())

        # 配置子图2 - 内存使用量
        self.line2, = self.ax2.plot([], [], 'g-', linewidth=2, label='已用')
        self.line3, = self.ax2.plot([], [], 'r-', linewidth=2, label='可用')
        self.ax2.set_ylabel('内存 (GB)')
        self.ax2.set_xlabel('时间')
        self.ax2.grid(True)
        self.ax2.legend()

        plt.tight_layout()

    def init(self):
        # 初始化线条
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.line3.set_data([], [])
        return self.line1, self.line2, self.line3

    def update(self, frame):
        # 获取当前时间
        current_time = datetime.datetime.now()

        # 获取内存信息
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024 ** 3)
        memory_available_gb = memory.available / (1024 ** 3)

        # 更新数据
        self.timestamps.append(current_time)
        self.memory_percent.append(memory.percent)
        self.memory_used.append(memory_used_gb)
        self.memory_available.append(memory_available_gb)

        # 限制数据点数量
        if len(self.timestamps) > MAX_POINTS:
            self.timestamps = self.timestamps[-MAX_POINTS:]
            self.memory_percent = self.memory_percent[-MAX_POINTS:]
            self.memory_used = self.memory_used[-MAX_POINTS:]
            self.memory_available = self.memory_available[-MAX_POINTS:]

        # 更新图表
        self.line1.set_data(self.timestamps, self.memory_percent)
        self.line2.set_data(self.timestamps, self.memory_used)
        self.line3.set_data(self.timestamps, self.memory_available)

        # 动态调整x轴范围
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        # 动态调整y轴
        self.ax2.set_ylim(0, self.total_memory * 1.1)

        # 控制台输出
        print(f"\r[{current_time.strftime('%H:%M:%S')}] "
              f"内存使用率: {memory.percent:.1f}% | "
              f"已用: {memory_used_gb:.2f} GB | "
              f"可用: {memory_available_gb:.2f} GB | "
              f"总计: {self.total_memory:.2f} GB", end="")

        # 警告高内存使用
        if memory.percent > ALERT_THRESHOLD and len(self.memory_percent) > 1:
            if self.memory_percent[-2] <= ALERT_THRESHOLD:  # 首次超过阈值
                print(f"\n⚠️ 警告: 内存使用率超过 {ALERT_THRESHOLD}%! 训练可能会崩溃。")

        return self.line1, self.line2, self.line3

    def run(self):
        ani = FuncAnimation(self.fig, self.update, frames=None,
                            init_func=self.init, blit=True, interval=UPDATE_INTERVAL * 1000)
        plt.show()


if __name__ == "__main__":
    print("ChatGLM训练内存监控工具")
    print("=" * 50)
    print(f"总内存: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    print(f"更新间隔: {UPDATE_INTERVAL}秒")
    print(f"警告阈值: {ALERT_THRESHOLD}%")
    print("=" * 50)
    print("开始监控... (按Ctrl+C退出)")

    try:
        monitor = MemoryMonitor()
        monitor.run()
    except KeyboardInterrupt:
        print("\n监控已停止")