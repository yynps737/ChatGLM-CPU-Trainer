# ChatGLM-CPU-Trainer

## 项目简介

**ChatGLM-CPU-Trainer** 是一个专为低资源环境设计的训练工具，让普通电脑也能训练和微调ChatGLM大模型。本项目使用Docker容器化部署，极大简化了环境配置难度。

## ⭐ 核心特性

- **CPU训练**：无需GPU，普通电脑也能训练
- **低内存优化**：从4GB到32GB内存均可运行
- **LoRA高效微调**：低参数量、高效率的微调方法
- **4-bit/8-bit量化**：大幅降低内存需求
- **Docker容器化**：一键部署，避免复杂环境配置

## 🖥️ 系统要求

- Docker引擎 (19.03+)
- 最低4GB内存（推荐8GB以上）
- Windows/Linux/macOS系统

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/ChatGLM-CPU-Trainer.git
cd ChatGLM-CPU-Trainer
```

### 2. 准备数据

将训练数据放入`data/input/dataset.txt`（每行一条训练样本）

### 3. 构建Docker镜像

```bash
docker build -t chatglm-cpu-trainer .
```

### 4. 训练模型

根据系统内存选择合适配置：

```bash
# 4GB内存
docker-compose -f docker-compose-4gb.yml run train

# 8GB内存
docker-compose -f docker-compose-8gb.yml run train

# 或使用默认配置
docker-compose run train
```

### 5. 测试模型

```bash
docker-compose run predict
```

## 📊 不同内存配置

| 内存 | 配置文件 | 量化 | 最大序列长度 | 样本数 |
|------|----------|------|------------|-------|
| 4GB  | docker-compose-4gb.yml | 4bit | 32 | 30 |
| 8GB  | docker-compose-8gb.yml | 4bit | 64 | 200 |
| 16GB | docker-compose-16gb.yml | 8bit | 128 | 800 |
| 32GB | docker-compose-32gb.yml | None | 256 | 2000 |

## ❓ 常见问题

- **内存不足**：选择更低内存配置或减少样本数
- **训练缓慢**：CPU训练本身较慢，耐心等待
- **模型下载失败**：检查网络，修改`HF_ENDPOINT`环境变量
- **量化失败**：尝试非量化模式（在内存允许的情况下）

## 📝 参数说明

主要参数说明：
- `--quantization`: 量化精度 (4bit/8bit/None)
- `--max_seq_length`: 最大序列长度，影响内存使用
- `--max_samples`: 训练样本数量
- `--lora_r`: LoRA秩，越小内存占用越低
- `--gradient_accumulation_steps`: 梯度累积步数，越大内存占用越低

## 📚 目录结构

```
ChatGLM-CPU-Trainer/
├── app/                    # 应用程序代码
├── data/                   # 数据目录
│   └── input/              # 输入数据
├── models/                 # 模型存储
├── scripts/                # 辅助脚本
├── Dockerfile              # Docker构建文件
└── docker-compose*.yml     # 不同配置文件
```