# ChatGLM-CPU-Trainer

纯CPU环境下的中文大模型训练框架，专为ChatGLM系列模型设计，支持低资源环境和消费级硬件。

## 特点

- 🖥️ **纯CPU训练**: 无需GPU，在普通PC和笔记本上也能运行
- ⚡ **低资源优化**: 量化 + LoRA + 内存优化，8GB内存也能使用
- 🇨🇳 **中文优先**: 针对ChatGLM等国产大模型优化的训练流程
- 📦 **简易部署**: 开箱即用，Windows/Linux全支持
- 🧩 **模块化设计**: 灵活组合，适应不同需求

## 快速开始

### Windows环境

```bash
# 1. 环境设置
scripts\setup_windows.bat

# 2. 开始训练
scripts\train_windows.bat

# 3. 简化版训练 (极低资源环境)
train_simple.bat
```

### Linux环境

```bash
# 1. 环境设置
chmod +x scripts/setup_linux.sh
./scripts/setup_linux.sh

# 2. 开始训练
chmod +x scripts/train_linux.sh
./scripts/train_linux.sh

# 3. 简化版训练 (极低资源环境)
python simple_train.py --use_lora --quantization 4bit --max_samples 1000
```

## 支持的模型

| 模型 | 内存需求 | 推荐序列长度 |
|-----|---------|------------|
| THUDM/chatglm3-6b | 8-16GB | 256 |
| THUDM/chatglm2-6b | 8-16GB | 256 |
| THUDM/chatglm3-6b-32k | 10-16GB | 512 |

## 微调示例

### 情感分析

```bash
# Windows
examples\sentiment_analysis.bat

# Linux
./examples/sentiment_analysis.sh
```

### 问答能力

```bash
# Windows
examples\qa_tuning.bat

# Linux
./examples/qa_tuning.sh
```

### 指令微调

```bash
# Windows
examples\instruction_tuning.bat

# Linux
./examples/instruction_tuning.sh
```

## 系统需求

- **操作系统**: Windows 10/11 或 Linux (Ubuntu 20.04+推荐)
- **Python**: 3.8-3.13
- **CPU**: 多核CPU (推荐8核以上)
- **内存**: 
  - 最低: 8GB (使用4bit量化, 限制样本数)
  - 推荐: 16GB
- **存储**: 10GB以上可用空间

## 优化技巧

### 内存优化

- **量化**: 使用`--quantization 4bit`替代`8bit`
- **序列长度**: 使用`--max_seq_length 128`减少序列长度
- **样本限制**: 使用`--max_samples 2000`限制数据量
- **LoRA参数**: 使用`--lora_r 4`减少适配器参数

### 速度优化

- **环境变量**: 设置`OMP_NUM_THREADS`和`MKL_NUM_THREADS`为CPU核心数
- **梯度累积**: 使用低批大小和高梯度累积步数
- **数据集大小**: 使用较小数据集进行测试

## 项目结构

```
chatglm-cpu-trainer/
├── scripts/                      # 脚本目录
├── src/                          # 源代码目录
│   ├── train/                    # 训练相关代码
│   ├── models/                   # 模型相关代码
│   └── utils/                    # 通用工具
├── configs/                      # 配置文件目录
├── examples/                     # 示例目录
├── simple_train.py               # 简化版训练脚本
├── train.py                      # 标准训练脚本
├── train_simple.bat              # Windows简化训练脚本
├── evaluate.py                   # 评估脚本
├── test_model.py                 # 模型测试脚本
├── memory_monitor.py             # 内存监控工具
├── requirements.txt              # 依赖列表
└── requirements_minimal.txt      # 最小依赖列表
```

## 常见问题

### 内存不足错误

**症状**: 训练过程中出现OOM错误

**解决方案**:
1. 使用简化版训练: `train_simple.bat` 或 `python simple_train.py`
2. 使用4bit量化: `--quantization 4bit`
3. 减小序列长度: `--max_seq_length 128`
4. 限制样本数量: `--max_samples 2000`
5. 减小LoRA秩: `--lora_r 4`

### 量化错误

**症状**: `Using load_in_8bit=True requires Accelerate`错误

**解决方案**:
```bash
# Windows
pip install -U accelerate bitsandbytes-windows

# Linux
pip install -U accelerate bitsandbytes
```

### 训练速度慢

**症状**: 训练一个批次耗时过长

**解决方案**:
1. 检查CPU线程设置: `OMP_NUM_THREADS=N`
2. 减小序列长度: `--max_seq_length 128`
3. 使用较小数据集: `--max_samples 1000`

## 协议

MIT License