"""
ChatGLM 训练脚本 - 专为Docker容器优化版本
"""

import os
import argparse
import torch
import gc
import time
import json
from datetime import datetime, timedelta
from transformers import set_seed, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
import platform
import logging
import signal
import sys

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
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA Dropout率")
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
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="学习率预热比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 优化器参数
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adafactor", "sgd"],
                        help="优化器类型")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam优化器beta1参数")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam优化器beta2参数")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam优化器epsilon参数")

    # 内存监控参数
    parser.add_argument("--monitor_memory", action="store_true", help="启用内存监控")
    parser.add_argument("--memory_check_interval", type=int, default=30,
                        help="内存检查间隔(秒)")

    # 性能监控参数
    parser.add_argument("--performance_log_steps", type=int, default=100,
                       help="每多少步记录一次性能指标")

    # 检查点设置
    parser.add_argument("--save_steps", type=int, default=500, help="保存检查点的步数间隔")
    parser.add_argument("--save_best_only", action="store_true", help="只保存最佳检查点")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存的检查点总数限制")

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
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )

class PerformanceTracker:
    """训练性能追踪器"""

    def __init__(self, logger, args):
        self.logger = logger
        self.args = args
        self.start_time = time.time()
        self.epoch_start_time = None
        self.step_start_time = None
        self.step_tokens = 0
        self.step_samples = 0
        self.total_tokens = 0
        self.total_samples = 0
        self.steps_taken = 0
        self.log_file = "/app/data/output/performance_metrics.csv"
        self.best_loss = float("inf")
        self.checkpoint_history = []  # 保存检查点历史

        # 创建性能日志文件
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,epoch,step,samples_per_second,tokens_per_second,loss,memory_used_mb,learning_rate\n")

        # 为训练历史创建JSON文件
        self.history_file = "/app/data/output/training_history.json"
        self.history = {
            "start_time": datetime.now().isoformat(),
            "args": vars(args),
            "system_info": self._get_system_info(),
            "epochs": []
        }
        self._save_history()

    def _get_system_info(self):
        """获取系统信息"""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "processor": platform.processor()
        }

        # 获取内存信息
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["total_memory_gb"] = round(mem.total / (1024**3), 2)
            info["available_memory_gb"] = round(mem.available / (1024**3), 2)
        except:
            pass

        # 获取PyTorch信息
        info["torch_version"] = torch.__version__

        return info

    def _save_history(self):
        """保存训练历史到JSON文件"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

    def start_epoch(self, epoch):
        """记录epoch开始时间"""
        self.epoch_start_time = time.time()
        self.logger.info(f"开始Epoch {epoch} 计时")

        # 创建新的epoch条目
        self.current_epoch = {
            "epoch": epoch,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "metrics": {}
        }

    def start_step(self, batch_size, sequence_length):
        """记录步骤开始时间和token数"""
        self.step_start_time = time.time()
        # 每个样本的token数量 = 序列长度
        self.step_tokens = batch_size * sequence_length
        self.step_samples = batch_size

    def end_step(self, epoch, global_step, loss, memory_used=None, learning_rate=None):
        """记录步骤结束和性能指标"""
        if self.step_start_time is None:
            return

        step_time = time.time() - self.step_start_time
        self.total_tokens += self.step_tokens
        self.total_samples += self.step_samples
        self.steps_taken += 1

        # 计算性能指标
        tokens_per_second = self.step_tokens / step_time if step_time > 0 else 0
        samples_per_second = self.step_samples / step_time if step_time > 0 else 0

        # 更新当前epoch步骤信息
        step_info = {
            "step": global_step,
            "loss": float(loss) if isinstance(loss, (int, float)) else None,
            "tokens_per_second": tokens_per_second,
            "samples_per_second": samples_per_second,
            "learning_rate": learning_rate
        }
        self.current_epoch["steps"].append(step_info)

        # 记录到CSV
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().isoformat()
            loss_str = float(loss) if isinstance(loss, (int, float)) else str(loss).replace(",", ";")
            lr_str = f"{learning_rate:.8f}" if learning_rate is not None else "N/A"
            f.write(f"{timestamp},{epoch},{global_step},{samples_per_second:.4f},{tokens_per_second:.4f},\"{loss_str}\",{memory_used or 0},{lr_str}\n")

        return {
            "tokens_per_second": tokens_per_second,
            "samples_per_second": samples_per_second
        }

    def end_epoch(self, epoch, epoch_loss=None):
        """记录epoch结束和汇总指标"""
        if self.epoch_start_time is None:
            return

        epoch_time = time.time() - self.epoch_start_time
        self.logger.info(f"Epoch {epoch} 完成，耗时: {timedelta(seconds=epoch_time)}")

        # 计算平均指标
        avg_tokens_per_second = self.total_tokens / epoch_time if epoch_time > 0 else 0
        avg_samples_per_second = self.total_samples / epoch_time if epoch_time > 0 else 0

        self.logger.info(f"Epoch {epoch} 平均性能: {avg_samples_per_second:.4f} 样本/秒, {avg_tokens_per_second:.4f} tokens/秒")

        if epoch_loss is not None:
            self.logger.info(f"Epoch {epoch} 平均损失: {epoch_loss:.6f}")

            # 更新最佳损失
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.logger.info(f"新的最佳损失: {self.best_loss:.6f}")
                return True  # 表示这是最佳模型

        # 更新epoch摘要信息
        self.current_epoch["end_time"] = datetime.now().isoformat()
        self.current_epoch["duration_seconds"] = epoch_time
        self.current_epoch["metrics"] = {
            "avg_tokens_per_second": avg_tokens_per_second,
            "avg_samples_per_second": avg_samples_per_second,
            "epoch_loss": epoch_loss
        }

        # 添加到历史记录
        self.history["epochs"].append(self.current_epoch)
        self._save_history()

        # 重置指标
        self.total_tokens = 0
        self.total_samples = 0
        self.steps_taken = 0

        return False  # 默认不是最佳模型

    def end_training(self, final_loss=None):
        """记录整体训练性能"""
        total_time = time.time() - self.start_time

        # 更新训练历史
        self.history["end_time"] = datetime.now().isoformat()
        self.history["duration_seconds"] = total_time
        self.history["final_loss"] = final_loss
        self.history["best_loss"] = self.best_loss
        self._save_history()

        self.logger.info(f"训练结束，总耗时: {timedelta(seconds=total_time)}")
        self.logger.info(f"最终损失: {final_loss:.6f}, 最佳损失: {self.best_loss:.6f}")
        self.logger.info(f"性能日志保存在: {self.log_file}")
        self.logger.info(f"训练历史保存在: {self.history_file}")

    def add_checkpoint(self, path, epoch, step, loss):
        """记录检查点"""
        checkpoint_info = {
            "path": path,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "timestamp": datetime.now().isoformat()
        }
        self.checkpoint_history.append(checkpoint_info)

        # 更新训练历史
        if "checkpoints" not in self.history:
            self.history["checkpoints"] = []
        self.history["checkpoints"].append(checkpoint_info)
        self._save_history()

        # 检查是否需要删除旧检查点
        self._manage_checkpoints()

    def _manage_checkpoints(self):
        """管理检查点，保持检查点数量在限制内"""
        if self.args.save_best_only:
            # 保存最佳检查点并删除其他所有检查点
            checkpoints = sorted(self.checkpoint_history, key=lambda x: x["loss"])
            best_checkpoint = checkpoints[0]

            for ckpt in checkpoints[1:]:
                try:
                    if os.path.exists(ckpt["path"]) and ckpt["path"] != best_checkpoint["path"]:
                        import shutil
                        shutil.rmtree(ckpt["path"])
                        self.logger.info(f"已删除非最佳检查点: {ckpt['path']}")
                except Exception as e:
                    self.logger.error(f"删除检查点出错: {e}")

        elif len(self.checkpoint_history) > self.args.save_total_limit > 0:
            # 保持检查点数量在限制内，删除最早的检查点
            checkpoints_to_delete = sorted(
                self.checkpoint_history,
                key=lambda x: x["timestamp"]
            )[:-self.args.save_total_limit]

            for ckpt in checkpoints_to_delete:
                try:
                    if os.path.exists(ckpt["path"]):
                        import shutil
                        shutil.rmtree(ckpt["path"])
                        self.logger.info(f"已删除旧检查点以保持数量在限制内: {ckpt['path']}")
                except Exception as e:
                    self.logger.error(f"删除检查点出错: {e}")

            # 更新检查点历史
            self.checkpoint_history = self.checkpoint_history[-self.args.save_total_limit:]

def setup_optimizer(model, args, num_training_steps):
    """设置优化器和学习率调度器"""
    # 获取需要优化的参数
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    # 选择合适的优化器
    if args.optimizer == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon
        )
    elif args.optimizer == "adafactor":
        from transformers import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            momentum=0.9
        )
    else:
        raise ValueError(f"不支持的优化器类型: {args.optimizer}")

    # 学习率调度器
    warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler

def train(args, model, tokenized_dataset, memory_monitor=None):
    """训练循环"""
    logger.info("开始训练...")

    # 检查数据集是否为空
    if len(tokenized_dataset) == 0:
        logger.error("数据集为空，无法开始训练")
        return model, None

    # 设置数据加载器
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True
    )

    # 检查数据加载器是否有数据
    if len(data_loader) == 0:
        logger.error("数据加载器为空，无法开始训练。可能是批大小过大或数据集为空。")
        return model, None

    # 计算总训练步数
    num_update_steps_per_epoch = len(data_loader) // args.gradient_accumulation_steps
    num_training_steps = num_update_steps_per_epoch * args.num_train_epochs

    # 设置优化器和学习率调度器
    optimizer, scheduler = setup_optimizer(model, args, num_training_steps)

    # 设置中断处理器
    stop_training = [False]
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def signal_handler(sig, frame):
        logger.warning("接收到中断信号，正在完成当前epoch后停止训练...")
        stop_training[0] = True
        # 恢复原始信号处理器，以便再次中断可以强制终止
        signal.signal(signal.SIGINT, original_sigint_handler)

    signal.signal(signal.SIGINT, signal_handler)

    # 开始训练循环
    model.train()

    # 初始化性能追踪器
    perf_tracker = PerformanceTracker(logger, args)

    # 如果启用了内存监控，开始监控
    if memory_monitor:
        memory_monitor.start_monitoring()
        # 训练开始时打印一次详细的内存信息
        memory_monitor.print_memory_info(detailed=True)

    final_loss = None

    try:
        for epoch in range(args.num_train_epochs):
            if stop_training[0]:
                logger.info("训练被用户中断，正在保存当前模型...")
                break

            logger.info(f"开始训练第 {epoch+1}/{args.num_train_epochs} 轮")
            running_loss = 0.0
            epoch_samples = 0

            # 开始记录这个epoch的性能
            perf_tracker.start_epoch(epoch + 1)

            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
            accumulated_loss = 0

            # 每个epoch开始时主动清理内存
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            for step, batch in enumerate(progress_bar):
                # 检查输入张量维度
                if len(batch['input_ids'].shape) != 2:
                    logger.warning(f"输入张量维度异常: {batch['input_ids'].shape}，预期为 [batch_size, seq_len]")
                    continue

                # 记录步骤开始时间和token数
                perf_tracker.start_step(
                    batch_size=len(batch['input_ids']),
                    sequence_length=batch['input_ids'].size(1)
                )

                epoch_samples += len(batch['input_ids'])

                # 将数据移动到设备上
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                # 前向传播
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                except RuntimeError as e:
                    logger.error(f"前向传播错误: {e}")
                    logger.info("跳过当前批次")
                    continue

                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()

                accumulated_loss += loss.item()

                # 梯度累积
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    running_loss += accumulated_loss
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': accumulated_loss * args.gradient_accumulation_steps,
                        'lr': f"{current_lr:.2e}"
                    })

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
                            memory_used=memory_used,
                            learning_rate=current_lr
                        )
                        # 更新进度条
                        if metrics:  # 确保metrics不为None
                            progress_bar.set_postfix({
                                'loss': accumulated_loss * args.gradient_accumulation_steps,
                                'samples/s': f"{metrics['samples_per_second']:.2f}",
                                'lr': f"{current_lr:.2e}"
                            })

                    accumulated_loss = 0

                    # 定期清理内存
                    if (step + 1) % (args.gradient_accumulation_steps * 10) == 0:
                        gc.collect()

                    # 检查点保存
                    if (step + 1) % (args.save_steps * args.gradient_accumulation_steps) == 0:
                        current_loss = running_loss / ((step + 1) // args.gradient_accumulation_steps)
                        logger.info(f"保存检查点 epoch {epoch+1}, step {step+1}, loss {current_loss:.6f}")
                        save_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}-step{step+1}")
                        os.makedirs(save_dir, exist_ok=True)

                        try:
                            model.save_pretrained(save_dir)
                            # 记录检查点
                            perf_tracker.add_checkpoint(save_dir, epoch+1, step+1, current_loss)
                            logger.info(f"检查点保存成功: {save_dir}")
                        except Exception as e:
                            logger.error(f"保存检查点失败: {str(e)}")

                        # 每个检查点打印一次当前内存使用情况
                        if memory_monitor:
                            memory_monitor.print_memory_info(detailed=True)
                            # 主动尝试垃圾回收
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 计算epoch的平均损失
            epoch_loss = running_loss / (len(data_loader) // args.gradient_accumulation_steps) if epoch_samples > 0 else float('inf')
            final_loss = epoch_loss
            logger.info(f"第 {epoch+1} 轮结束, 平均损失: {epoch_loss:.6f}")

            # 结束当前epoch的性能记录，判断是否是最佳模型
            is_best = perf_tracker.end_epoch(epoch + 1, epoch_loss)

            # 保存每个epoch的模型
            epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)

            try:
                model.save_pretrained(epoch_dir)
                # 记录检查点
                perf_tracker.add_checkpoint(epoch_dir, epoch+1, len(data_loader), epoch_loss)
                logger.info(f"Epoch {epoch+1} 模型保存成功: {epoch_dir}")

                # 如果是最佳模型，保存到best目录
                if is_best:
                    best_dir = os.path.join(args.output_dir, "best")
                    os.makedirs(best_dir, exist_ok=True)
                    model.save_pretrained(best_dir)
                    logger.info(f"保存最佳模型到: {best_dir}")
            except Exception as e:
                logger.error(f"保存Epoch {epoch+1} 模型失败: {str(e)}")

            # 每个epoch结束后主动清理内存
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 每个epoch结束后打印详细内存信息
            if memory_monitor:
                memory_monitor.print_memory_info(detailed=True)

    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        # 尝试保存当前模型
        try:
            emergency_dir = os.path.join(args.output_dir, "emergency-save")
            os.makedirs(emergency_dir, exist_ok=True)
            model.save_pretrained(emergency_dir)
            logger.info(f"模型已紧急保存到: {emergency_dir}")
        except:
            logger.error("无法进行紧急保存")
    finally:
        # 停止内存监控
        if memory_monitor:
            memory_monitor.stop_monitoring()

        # 结束性能记录
        perf_tracker.end_training(final_loss)

        # 恢复原始信号处理器
        signal.signal(signal.SIGINT, original_sigint_handler)

    return model, final_loss

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
    logger.info(f"LoRA参数: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"序列长度: {args.max_seq_length}, 样本数: {args.max_samples}")
    logger.info(f"批大小: {args.per_device_train_batch_size}, 梯度累积: {args.gradient_accumulation_steps}")
    logger.info(f"学习率: {args.learning_rate}, 权重衰减: {args.weight_decay}")
    logger.info(f"优化器: {args.optimizer}")
    logger.info("=" * 50)

    # 检查文件是否存在
    if not os.path.exists(args.dataset_path):
        logger.error(f"数据集文件不存在: {args.dataset_path}")
        return

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

    try:
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
        model, final_loss = train(args, model, tokenized_dataset, memory_monitor)

        # 保存最终模型
        logger.info(f"保存模型到 {args.output_dir}")
        try:
            model.save_pretrained(args.output_dir)  # 只保存LoRA权重
            tokenizer.save_pretrained(args.output_dir)  # 保存分词器

            # 保存训练元数据
            with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)

            # 保存最终损失
            with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
                json.dump({"final_loss": final_loss}, f, indent=2)

            logger.info("模型和分词器保存成功")
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")

    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    # 计算总训练时间
    total_time = time.time() - start_time

    logger.info("=" * 50)
    logger.info("训练完成!")
    logger.info(f"模型已保存到: {args.output_dir}")
    logger.info(f"总训练时间: {timedelta(seconds=total_time)}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()