"""
ChatGLM 训练脚本 - 专为Docker容器优化版本
"""

import os
import argparse
import torch
import gc
import time
from datetime import datetime, timedelta
from transformers import set_seed
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
import platform

# 导入工具函数
from app.utils import setup_logging, load_model_and_tokenizer, prepare_dataset
from app.memory_monitor import MemoryMonitor  # 导入内存监控器

# 设置日志
logger = setup_logging(log_file="/app/data/output/train.log")

# 禁用CUDA，仅使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ChatGLM训练脚本 (Docker优化版)")

    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm2-6b",
                        help="模型名称或路径")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA注意力维度")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha参数")
    parser.add_argument("--quantization", type=str, default="None",
                        choices=["4bit", "8bit", "None"],
                        help="模型量化类型")

    # 数据参数
    parser.add_argument("--dataset_path", type=str, default="/app/data/input/dataset.txt",
                    help="本地数据集路径，支持.csv, .json, .jsonl, .txt格式")
    parser.add_argument("--text_column", type=str, default="text", help="文本列名称")
    parser.add_argument("--max_seq_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=1000, help="要使用的最大样本数")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="/app/models/chatglm-lora",
                        help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每个设备的训练批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 内存监控参数
    parser.add_argument("--monitor_memory", action="store_true", help="启用内存监控")
    parser.add_argument("--memory_check_interval", type=int, default=30,
                        help="内存检查间隔(秒)")

    # 性能监控参数
    parser.add_argument("--performance_log_steps", type=int, default=100,
                       help="每多少步记录一次性能指标")

    return parser.parse_args()

def create_lora_config(args):
    """创建LoRA配置"""
    # 为ChatGLM模型定制的目标模块
    target_modules = [
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
    ]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )

class PerformanceTracker:
    """训练性能追踪器"""

    def __init__(self, logger):
        self.logger = logger
        self.start_time = time.time()
        self.epoch_start_time = None
        self.step_start_time = None
        self.step_tokens = 0
        self.total_tokens = 0
        self.total_samples = 0
        self.steps_taken = 0
        self.log_file = "/app/data/output/performance_metrics.csv"

        # 创建性能日志文件
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            f.write("timestamp,epoch,step,samples_per_second,tokens_per_second,loss,memory_used_mb\n")

    def start_epoch(self, epoch):
        """记录epoch开始时间"""
        self.epoch_start_time = time.time()
        self.logger.info(f"开始Epoch {epoch} 计时")

    def start_step(self, batch_size, sequence_length):
        """记录步骤开始时间和token数"""
        self.step_start_time = time.time()
        # 每个样本的token数量 = 序列长度
        self.step_tokens = batch_size * sequence_length

    def end_step(self, epoch, global_step, loss, memory_used=None):
        """记录步骤结束和性能指标"""
        if self.step_start_time is None:
            return

        step_time = time.time() - self.step_start_time
        self.total_tokens += self.step_tokens
        self.total_samples += self.step_tokens // self.step_tokens
        self.steps_taken += 1

        # 计算性能指标
        tokens_per_second = self.step_tokens / step_time if step_time > 0 else 0
        samples_per_second = (self.step_tokens // self.step_tokens) / step_time if step_time > 0 else 0

        # 记录到CSV
        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp},{epoch},{global_step},{samples_per_second:.4f},{tokens_per_second:.4f},{loss},{memory_used or 0}\n")

        return {
            "tokens_per_second": tokens_per_second,
            "samples_per_second": samples_per_second
        }

    def end_epoch(self, epoch):
        """记录epoch结束和汇总指标"""
        if self.epoch_start_time is None:
            return

        epoch_time = time.time() - self.epoch_start_time
        self.logger.info(f"Epoch {epoch} 完成，耗时: {timedelta(seconds=epoch_time)}")

        # 计算平均指标
        avg_tokens_per_second = self.total_tokens / epoch_time if epoch_time > 0 else 0
        avg_samples_per_second = self.total_samples / epoch_time if epoch_time > 0 else 0

        self.logger.info(f"Epoch {epoch} 平均性能: {avg_samples_per_second:.4f} 样本/秒, {avg_tokens_per_second:.4f} tokens/秒")

        # 重置指标
        self.total_tokens = 0
        self.total_samples = 0
        self.steps_taken = 0

    def end_training(self):
        """记录整体训练性能"""
        total_time = time.time() - self.start_time
        self.logger.info(f"训练结束，总耗时: {timedelta(seconds=total_time)}")
        self.logger.info(f"性能日志保存在: {self.log_file}")

def train(args, model, tokenized_dataset, memory_monitor=None):
    """训练循环"""
    logger.info("开始训练...")

    # 设置数据加载器
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True
    )

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 开始训练循环
    model.train()

    # 初始化性能追踪器
    perf_tracker = PerformanceTracker(logger)

    # 如果启用了内存监控，开始监控
    if memory_monitor:
        memory_monitor.start_monitoring()
        # 训练开始时打印一次详细的内存信息
        memory_monitor.print_memory_info(detailed=True)

    try:
        for epoch in range(args.num_train_epochs):
            logger.info(f"开始训练第 {epoch+1}/{args.num_train_epochs} 轮")
            running_loss = 0.0

            # 开始记录这个epoch的性能
            perf_tracker.start_epoch(epoch + 1)

            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
            accumulated_loss = 0

            # 每个epoch开始时主动清理内存
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            for step, batch in enumerate(progress_bar):
                # 记录步骤开始时间和token数
                perf_tracker.start_step(
                    batch_size=len(batch['input_ids']),
                    sequence_length=batch['input_ids'].size(1)
                )

                # 将数据移动到设备上
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()

                accumulated_loss += loss.item()

                # 梯度累积
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    running_loss += accumulated_loss
                    progress_bar.set_postfix({'loss': accumulated_loss * args.gradient_accumulation_steps})

                    # 获取当前内存使用情况
                    memory_used = None
                    if memory_monitor:
                        mem_info = memory_monitor.get_memory_info()
                        memory_used = mem_info['process']['rss'] / (1024 * 1024)  # MB

                    # 记录性能指标
                    global_step = epoch * len(data_loader) + step
                    if global_step % args.performance_log_steps == 0:
                        metrics = perf_tracker.end_step(
                            epoch=epoch+1,
                            global_step=global_step,
                            loss=accumulated_loss * args.gradient_accumulation_steps,
                            memory_used=memory_used
                        )
                        # 更新进度条
                        progress_bar.set_postfix({
                            'loss': accumulated_loss * args.gradient_accumulation_steps,
                            'samples/s': f"{metrics['samples_per_second']:.2f}"
                        })

                    accumulated_loss = 0

                    # 定期清理内存
                    if (step + 1) % (args.gradient_accumulation_steps * 10) == 0:
                        gc.collect()

                    # 检查点保存
                    if (step + 1) % (args.gradient_accumulation_steps * 50) == 0:
                        logger.info(f"保存检查点 epoch {epoch+1}, step {step+1}")
                        save_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}-step{step+1}")
                        os.makedirs(save_dir, exist_ok=True)
                        model.save_pretrained(save_dir)

                        # 每个检查点打印一次当前内存使用情况
                        if memory_monitor:
                            memory_monitor.print_memory_info(detailed=True)
                            # 主动尝试垃圾回收
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 结束当前epoch的性能记录
            perf_tracker.end_epoch(epoch + 1)

            # 保存每个epoch的模型
            epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)

            logger.info(f"第 {epoch+1} 轮结束, 平均损失: {running_loss / len(data_loader)}")

            # 每个epoch结束后主动清理内存
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 每个epoch结束后打印详细内存信息
            if memory_monitor:
                memory_monitor.print_memory_info(detailed=True)

    finally:
        # 停止内存监控
        if memory_monitor:
            memory_monitor.stop_monitoring()

        # 结束性能记录
        perf_tracker.end_training()

    return model

def main():
    """主函数"""
    # 记录开始时间
    start_time = time.time()

    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置CPU环境变量
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "4")

    logger.info("=" * 50)
    logger.info("ChatGLM CPU 训练脚本 (Docker优化版)")
    logger.info("=" * 50)
    logger.info(f"模型: {args.model_name_or_path}")
    logger.info(f"本地数据集: {args.dataset_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"量化级别: {args.quantization}")
    logger.info(f"LoRA参数: r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"序列长度: {args.max_seq_length}, 样本数: {args.max_samples}")
    logger.info(f"批大小: {args.per_device_train_batch_size}, 梯度累积: {args.gradient_accumulation_steps}")
    logger.info("=" * 50)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化内存监控
    memory_monitor = None
    if args.monitor_memory:
        logger.info(f"启用内存监控 (检查间隔: {args.memory_check_interval}秒)")
        memory_monitor = MemoryMonitor(
            logger=logger,
            check_interval=args.memory_check_interval
        )

    # 主动清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.quantization)

    # 添加LoRA适配器
    logger.info("添加LoRA适配器...")
    peft_config = create_lora_config(args)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 准备数据集前再次清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 准备数据集
    tokenized_dataset = prepare_dataset(
        args.dataset_path,
        args.text_column,
        args.max_samples,
        tokenizer,
        args.max_seq_length
    )

    # 训练模型
    model = train(args, model, tokenized_dataset, memory_monitor)

    # 保存最终模型
    logger.info(f"保存模型到 {args.output_dir}")
    model.save_pretrained(args.output_dir)  # 只保存LoRA权重
    tokenizer.save_pretrained(args.output_dir)  # 保存分词器

    # 计算总训练时间
    total_time = time.time() - start_time

    logger.info("=" * 50)
    logger.info("训练完成!")
    logger.info(f"模型已保存到: {args.output_dir}")
    logger.info(f"总训练时间: {timedelta(seconds=total_time)}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()