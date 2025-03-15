# ChatGLM-CPU-Trainer

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![许可](https://img.shields.io/badge/许可-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8+-orange)
![Docker](https://img.shields.io/badge/Docker-支持-brightgreen)

ChatGLM-CPU-Trainer 是一个专为资源受限环境设计的 ChatGLM 模型训练与微调工具，让普通电脑也能进行大语言模型微调。

## 📑 目录

- [核心特性](#-核心特性)
- [技术架构](#-技术架构)
- [系统要求](#-系统要求)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
- [高级配置](#-高级配置)
- [项目结构](#-项目结构)
- [常见问题](#-常见问题)
- [贡献指南](#-贡献指南)
- [致谢](#-致谢)

## 🔑 核心特性

- **低资源训练**: 支持在普通PC（8GB内存）上微调大语言模型
- **量化支持**: 提供4-bit和8-bit量化选项，大幅降低内存占用（最高可减少87.5%）
- **LoRA微调**: 使用参数高效微调技术，显著减少需要训练的参数量
- **自适应配置**: 自动检测系统资源并应用最佳配置参数
- **内存监控**: 实时跟踪内存使用，防止OOM错误
- **Docker封装**: 简化环境配置，确保跨平台兼容性
- **统一接口**: 通过Makefile提供简单一致的命令接口

## 🔧 技术架构

本项目基于以下核心技术:

- **PEFT (LoRA)**: 参数高效微调技术，只训练少量适配器参数
- **量化技术**: 利用bitsandbytes库实现4-bit和8-bit模型量化
- **Docker容器化**: 提供一致的运行环境，避免依赖问题
- **内存优化**: 针对CPU环境的特殊内存管理和优化策略
- **自动配置**: 基于系统资源自动选择最佳参数配置

## 💻 系统要求

- **操作系统**: Linux, macOS, 或 Windows (需支持Docker)
- **内存**: 最低8GB (推荐16GB或以上)
- **存储**: 20GB可用磁盘空间
- **软件依赖**:
  - Docker Engine
  - Docker Compose
  - Make (用于命令执行)
- **网络连接**: 用于模型和依赖下载

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/ChatGLM-CPU-Trainer.git
cd ChatGLM-CPU-Trainer

# 创建必要目录结构并构建Docker镜像
make setup
```

### 训练模型

```bash
# 使用默认配置训练
make train

# 或使用自定义参数
make train MAX_SAMPLES=100 NUM_EPOCHS=5
```

### 使用模型生成文本

```bash
# 使用默认提示词
make predict

# 使用自定义提示词
make predict-custom PROMPT="请解释量子计算的基本原理"
```

## 📘 使用指南

### 准备训练数据

将训练数据放入`data/input/dataset.txt`文件中，每行一个样本。支持的文件格式:
- `.txt`: 每行一个样本
- `.csv`: 带标题行的CSV文件
- `.json`/`.jsonl`: JSON或JSON Lines格式

### 训练参数

您可以通过`make train`命令后追加参数来自定义训练过程:

```bash
make train MAX_SAMPLES=50 NUM_EPOCHS=3 BATCH_SIZE=1 GRAD_ACCUM=16
```

主要参数说明:
- `MAX_SAMPLES`: 使用的最大样本数
- `NUM_EPOCHS`: 训练轮数
- `BATCH_SIZE`: 批处理大小
- `GRAD_ACCUM`: 梯度累积步数
- `QUANT_LEVEL`: 量化级别 (4bit, 8bit, None)

### 模型推理

训练完成后，模型权重将保存在`models/chatglm-lora`目录中。使用以下命令测试模型:

```bash
# 默认提示词
make predict

# 自定义提示词
make predict-custom PROMPT="请介绍一下自然语言处理的发展历史"
```

## ⚙️ 高级配置

### 内存配置详解

系统会根据检测到的内存自动选择最佳配置:

| 内存配置 | 系统内存 | 量化方式 | 序列长度 | 样本数 | 批大小 | 梯度累积 |
|----------|---------|---------|---------|--------|--------|---------|
| 4GB      | <6GB    | 4-bit   | 32      | 30     | 1      | 32      |
| 8GB      | <12GB   | 4-bit   | 64      | 200    | 1      | 16      |
| 16GB     | <24GB   | 8-bit   | 128     | 800    | 2      | 8       |
| 32GB     | ≥24GB   | 无量化   | 256     | 2000   | 4      | 4       |

### 量化选项说明

- **4-bit量化**: 内存占用最小，约为原始模型的1/8，适合低内存环境
- **8-bit量化**: 内存占用适中，约为原始模型的1/4，平衡精度和内存需求
- **无量化**: 内存占用最大，但保持完整精度，适合高内存环境

### 直接使用Docker命令

如果您熟悉Docker，也可以直接使用Docker命令:

```bash
# 配置环境
docker-compose run --rm setup

# 训练模型
docker-compose run --rm -e MAX_SAMPLES=100 train

# 预测
docker-compose run --rm -e PROMPT="你的提示词" predict
```

## 📁 项目结构

```
ChatGLM-CPU-Trainer/
├── app/                      # 核心代码
│   ├── __init__.py           # 包初始化
│   ├── memory_monitor.py     # 内存监控模块
│   ├── predict.py            # 预测脚本
│   ├── quantization.py       # 模型量化工具
│   ├── train.py              # 训练脚本
│   └── utils.py              # 工具函数
├── data/                     # 数据目录
│   ├── input/                # 训练数据
│   └── output/               # 输出和日志
├── models/                   # 模型存储目录
├── .dockerignore             # Docker忽略文件
├── .env.template             # 环境变量模板
├── Dockerfile                # Docker配置文件
├── docker-compose.yml        # Docker Compose配置
├── Makefile                  # Make命令配置
└── README.md                 # 项目说明文档
```

## ❓ 常见问题

### 模型下载速度慢

如果遇到模型下载缓慢的问题，可以尝试:
1. 在`.env`文件中修改镜像站：`HF_ENDPOINT=https://hf-mirror.com`
2. 手动下载模型并放入缓存目录：`~/.cache/huggingface`

### 内存不足错误

如果训练过程中遇到内存不足错误:
1. 减少`MAX_SAMPLES`或`MAX_SEQ_LEN`参数
2. 使用更高级别的量化（从无量化切换到8-bit或4-bit）
3. 增加梯度累积步数`GRAD_ACCUM`
4. 确保关闭其他内存占用大的应用

### Docker相关问题

如果遇到Docker相关问题:
1. 确保Docker和Docker Compose已正确安装并运行
2. 检查命令行输出以获取详细错误信息
3. 尝试重启Docker服务
4. 使用`docker system prune`清理未使用的资源

### 训练速度很慢

CPU训练本身比GPU慢得多，但您可以通过以下方法加快速度:
1. 减少`MAX_SEQ_LEN`和`MAX_SAMPLES`
2. 减少训练轮数`NUM_EPOCHS`
3. 在多核CPU机器上，增加`NUM_THREADS`参数
4. 使用数据子集快速验证模型配置，然后再使用完整数据集

## 🤝 贡献指南

我们欢迎各种形式的贡献，包括但不限于:

- 代码优化和Bug修复
- 新功能开发
- 文档改进和翻译
- 使用案例和教程

要贡献代码，请:
1. Fork本仓库
2. 创建您的功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

## 📜 致谢

- ChatGLM模型由清华大学开发
- 使用了Hugging Face的Transformers、PEFT和bitsandbytes库
- 内存优化和量化技术借鉴了多个开源项目的最佳实践
- 感谢所有直接或间接为本项目做出贡献的开发者