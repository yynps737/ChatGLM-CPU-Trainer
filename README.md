# ChatGLM-CPU-Trainer

## 项目简介

**ChatGLM-CPU-Trainer** 是一个专为低资源环境设计的训练工具，让普通电脑也能训练和微调ChatGLM大模型。本项目使用Docker容器化部署，极大简化了环境配置难度。

## ⭐ 核心特性

- **CPU训练**：无需GPU，普通电脑也能训练
- **低内存优化**：从4GB到32GB内存均可运行
- **LoRA高效微调**：低参数量、高效率的微调方法
- **4-bit/8-bit量化**：大幅降低内存需求
- **Docker容器化**：一键部署，避免复杂环境配置
- **自动配置**：根据系统内存自动选择最优配置

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

### 2. 运行自动设置脚本

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```
该脚本会自动检测您的系统内存并选择最合适的配置设置。

### 3. 准备数据

将训练数据放入`data/input/dataset.txt`（每行一条训练样本）

### 4. 构建Docker镜像

```bash
docker build -t chatglm-cpu-trainer .
```

### 5. 训练模型

```bash
# 使用根据您系统内存自动配置的参数
docker-compose run train
```

### 6. 测试模型

```bash
docker-compose run predict
```

## 📊 内存配置

项目通过`.env`文件管理不同的内存配置。`scripts/setup.sh`会自动选择合适的配置，您也可以手动修改：

| 内存 | 配置名称 | 量化 | 最大序列长度 | 样本数 |
|------|----------|------|------------|-------|
| 4GB  | 4gb | 4bit | 32 | 30 |
| 8GB  | 8gb | 4bit | 64 | 200 |
| 16GB | 16gb | 8bit | 128 | 800 |
| 32GB | 32gb | None | 256 | 2000 |

要手动切换配置，只需编辑`.env`文件，取消注释对应的配置部分即可。

## ⚙️ 自定义参数

您可以通过环境变量自定义训练参数，例如：

```bash
# 修改序列长度和样本数
MEMORY_CONFIG=8gb MAX_SEQ_LEN=128 MAX_SAMPLES=400 docker-compose run train

# 使用不同的提示进行预测
PROMPT="请简要介绍机器学习的基本原理。" docker-compose run predict
```

## ❓ 常见问题

- **内存不足**：运行`scripts/setup.sh`选择更低内存配置，或手动编辑`.env`文件
- **训练缓慢**：CPU训练本身较慢，耐心等待或考虑减少`MAX_SAMPLES`参数
- **模型下载失败**：检查网络，修改`HF_ENDPOINT`环境变量
- **量化失败**：尝试重启训练；系统会自动回退到非量化模式（会占用更多内存）

## 📝 参数说明

主要参数说明（可在`.env`文件中配置）：
- `QUANT_LEVEL`: 量化精度 (4bit/8bit/None)
- `MAX_SEQ_LEN`: 最大序列长度，影响内存使用
- `MAX_SAMPLES`: 训练样本数量
- `LORA_R`: LoRA秩，越小内存占用越低
- `GRAD_ACCUM`: 梯度累积步数，越大内存占用越低

## 📚 目录结构

```
ChatGLM-CPU-Trainer/
├── app/                    # 应用程序代码
│   ├── __init__.py         # 包初始化
│   ├── predict.py          # 预测脚本
│   ├── quantization.py     # 量化工具
│   ├── train.py            # 训练脚本
│   └── utils.py            # 通用工具函数
├── data/                   # 数据目录
│   └── input/              # 输入数据
├── models/                 # 模型存储
├── scripts/                # 辅助脚本
│   ├── predict.sh          # 预测辅助脚本
│   ├── setup.sh            # 自动配置脚本
│   └── train.sh            # 训练辅助脚本
├── .env                    # 环境配置
├── .env.example            # 环境配置示例
├── Dockerfile              # Docker构建文件
├── docker-compose.yml      # Docker Compose配置
└── requirements.txt        # 依赖列表
```