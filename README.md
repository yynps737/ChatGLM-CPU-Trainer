# ChatGLM-CPU-Trainer

## 项目介绍

ChatGLM-CPU-Trainer 是一个专为低资源环境设计的 ChatGLM 模型训练工具，特别优化用于在 CPU 上进行高效训练和微调。该项目旨在让没有高性能 GPU 的用户也能训练和定制自己的 ChatGLM 模型。

## 功能特点

- **CPU 优化**：专为 CPU 环境设计，无需 GPU 即可训练
- **低资源配置**：针对内存和计算资源有限的设备进行优化
- **LoRA 高效微调**：使用 LoRA（Low-Rank Adaptation）技术进行高效参数微调
- **模型量化支持**：支持 4bit/8bit 量化

## 系统要求

- Python 3.10 或更高版本
- 至少 8GB RAM（推荐 16GB 以上）
- Ubuntu 或 Windows 操作系统

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/ChatGLM-CPU-Trainer.git
cd ChatGLM-CPU-Trainer
```

### 2. 安装依赖

```bash
# 使用最小化依赖安装
pip install -r requirements_minimal.txt
```

## 使用方法

### Windows 用户快速开始

直接运行批处理文件进行训练：

```
train_simple.bat
```

### 手动训练

```bash
python simple_train.py \
  --model_name_or_path THUDM/chatglm2-6b \
  --dataset_name uer/cluecorpussmall \
  --use_lora \
  --lora_r 8 \
  --quantization 4bit \
  --max_seq_length 128 \
  --max_samples 1000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --output_dir output/chatglm-lora
```

### 测试训练好的模型

```bash
python test_model.py \
  --model_path output/chatglm-lora \
  --base_model_path THUDM/chatglm2-6b \
  --quantization 4bit \
  --prompt "请介绍一下人工智能的发展历史。"
```

## 参数说明

### 训练参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--model_name_or_path` | 模型名称或路径 | `THUDM/chatglm2-6b` |
| `--lora_r` | LoRA 注意力维度 | `8` |
| `--lora_alpha` | LoRA Alpha 参数 | `32` |
| `--quantization` | 模型量化类型 (4bit, 8bit, None) | `None` |
| `--dataset_name` | Hugging Face 数据集名称 | `uer/cluecorpussmall` |
| `--text_column` | 文本列名称 | `text` |
| `--max_seq_length` | 最大序列长度 | `128` |
| `--max_samples` | 要使用的最大样本数 | `1000` |
| `--output_dir` | 输出目录 | `./output/chatglm-lora` |
| `--num_train_epochs` | 训练轮数 | `3` |
| `--per_device_train_batch_size` | 每个设备的训练批大小 | `1` |
| `--gradient_accumulation_steps` | 梯度累积步数 | `16` |
| `--learning_rate` | 学习率 | `5e-5` |
| `--seed` | 随机种子 | `42` |

### 测试参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--model_path` | LoRA 模型路径 | 必填 |
| `--base_model_path` | 基础模型路径 | 必填 |
| `--prompt` | 测试提示 | `请介绍一下人工智能的发展历史。` |
| `--quantization` | 模型量化类型 (4bit, 8bit, None) | `None` |
| `--max_length` | 生成的最大长度 | `2048` |

## 低资源训练技巧

1. **使用 4bit 量化**：减少内存使用，但可能影响生成质量
2. **减少序列长度**：降低 `max_seq_length` 参数值
3. **减少训练样本数**：通过 `max_samples` 参数限制样本数量
4. **降低 LoRA 秩**：减小 `lora_r` 参数值可以减少训练参数数量
5. **增加梯度累积步数**：可以使用更小的批处理大小

## 常见问题

### 内存不足

如果出现 OOM (Out of Memory) 错误，尝试：

1. 减少 `max_samples` 和 `max_seq_length`
2. 使用 4bit 量化
3. 减小 `lora_r` 值

### BitsAndBytes 安装问题（Windows）

Windows 用户如果遇到 bitsandbytes 相关错误，尝试安装特定版本：

```
pip uninstall bitsandbytes-windows
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```
反馈邮箱：ysa@kiki20.com
## 许可证

暂无