# ChatGLM-CPU-Trainer

## 项目简介

**ChatGLM-CPU-Trainer** 是一个专为低资源环境设计的训练工具，让普通电脑也能训练和微调ChatGLM大模型。本项目使用Docker容器化部署，极大简化了环境配置难度。无需GPU，即使是配置较低的普通电脑也能实现模型训练与微调。

## ⭐ 核心特性

- **CPU训练**：无需GPU，普通电脑也能训练
- **低内存优化**：从4GB到32GB内存均可运行
- **LoRA高效微调**：低参数量、高效率的微调方法
- **4-bit/8-bit量化**：大幅降低内存需求
- **Docker容器化**：一键部署，避免复杂环境配置
- **自动配置**：根据系统内存自动选择最优配置
- **内存监控**：实时监控系统资源使用情况，防止内存溢出
- **性能追踪**：记录训练过程中的性能指标，便于优化调整
- **Windows支持**：提供Windows专用批处理脚本，一键操作

## 🖥️ 系统要求

- Docker引擎 (19.03+)
- 最低4GB内存（推荐8GB以上）
- Windows/Linux/macOS系统
- 互联网连接（用于下载模型）

## 🚀 快速开始（Windows）

### 1. 安装Docker Desktop

首先确保您已安装Docker Desktop for Windows。如果尚未安装，请访问[Docker官网](https://www.docker.com/products/docker-desktop/)下载并安装。

### 2. 克隆或下载本仓库

```
git clone https://github.com/your-username/ChatGLM-CPU-Trainer.git
```
或直接从GitHub下载ZIP压缩包并解压缩。

### 3. 初始化环境

双击运行项目根目录下的`scripts\setup.bat`脚本，它将自动：
- 检测系统内存并选择最佳配置
- 创建必要的目录结构
- 生成.env配置文件

### 4. 准备训练数据

将训练数据放入`data\input\dataset.txt`文件中（每行一条训练样本）。

### 5. 一键训练

双击运行项目根目录下的`一键训练.bat`。首次运行时会自动构建Docker镜像，之后开始训练过程。

### 6. 一键预测

训练完成后，双击运行`一键预测.bat`，输入您想要的提示词，即可查看模型回复。

## 🚀 快速开始（Linux/macOS）

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

将训练数据放入`data/input/dataset.txt`（每行一条训练样本）。支持以下格式：
- `.txt`：每行一个样本
- `.csv`：包含文本列的CSV文件
- `.json`/`.jsonl`：包含文本字段的JSON/JSONL文件

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
# 使用默认提示进行测试
docker-compose run predict

# 或使用自定义提示
PROMPT="请简要介绍机器学习的基本原理。" docker-compose run predict
```

## 📊 内存配置

项目通过`.env`文件管理不同的内存配置。`setup`脚本会自动选择合适的配置，您也可以手动修改：

| 内存 | 配置名称 | 量化 | 最大序列长度 | 样本数 |
|------|----------|------|------------|-------|
| 4GB  | 4gb | 4bit | 32 | 30 |
| 8GB  | 8gb | 4bit | 64 | 200 |
| 16GB | 16gb | 8bit | 128 | 800 |
| 32GB | 32gb | None | 256 | 2000 |

## ⚙️ 自定义参数

### Windows环境下的自定义参数

您可以编辑.env文件，修改各项参数。也可以通过scripts目录下的详细脚本执行更自定义的操作：

```
scripts\train.bat --help  # 查看训练参数帮助
scripts\predict.bat --help  # 查看预测参数帮助
```

### Linux/macOS环境下的自定义参数

```bash
# 修改序列长度和样本数
MEMORY_CONFIG=8gb MAX_SEQ_LEN=128 MAX_SAMPLES=400 docker-compose run train

# 使用不同的提示进行预测
PROMPT="请简要介绍机器学习的基本原理。" docker-compose run predict

# 修改模型路径
MODEL="THUDM/chatglm3-6b" docker-compose run train
```

## 💻 命令行工具

### Windows批处理脚本

- **项目根目录**
  - `一键训练.bat` - 一键执行训练过程
  - `一键预测.bat` - 一键执行预测过程

- **scripts目录**
  - `setup.bat` - 初始化环境与配置
  - `train.bat` - 详细训练脚本，支持参数自定义
  - `predict.bat` - 详细预测脚本，支持参数自定义
  
### Linux/macOS Shell脚本

- `scripts/setup.sh` - 环境配置脚本
- `scripts/train.sh` - 训练脚本
- `scripts/predict.sh` - 预测脚本

## 📊 性能监控

训练过程中会自动记录内存使用情况和性能指标，结果保存在：
- 内存监控日志：`data/output/memory_usage.csv`
- 性能指标日志：`data/output/performance_metrics.csv`

## 📝 参数说明

主要参数说明（可在`.env`文件中配置）：
- `QUANT_LEVEL`: 量化精度 (4bit/8bit/None)
- `MAX_SEQ_LEN`: 最大序列长度，影响内存使用
- `MAX_SAMPLES`: 训练样本数量
- `LORA_R`: LoRA秩，越小内存占用越低
- `GRAD_ACCUM`: 梯度累积步数，越大内存占用越低
- `BATCH_SIZE`: 批处理大小，通常保持为1
- `NUM_THREADS`: CPU线程数

## 📚 目录结构

```
ChatGLM-CPU-Trainer/
├── app/                    # 应用程序代码
│   ├── __init__.py         # 包初始化
│   ├── memory_monitor.py   # 内存监控工具
│   ├── predict.py          # 预测脚本
│   ├── quantization.py     # 量化工具
│   ├── train.py            # 训练脚本
│   └── utils.py            # 通用工具函数
├── data/                   # 数据目录
│   ├── input/              # 输入数据
│   └── output/             # 输出结果和日志
├── models/                 # 模型存储
├── scripts/                # 辅助脚本目录
│   ├── predict.bat         # Windows预测脚本
│   ├── predict.sh          # Linux预测脚本
│   ├── setup.bat           # Windows配置脚本
│   ├── setup.sh            # Linux配置脚本
│   ├── train.bat           # Windows训练脚本
│   └── train.sh            # Linux训练脚本
├── .env                    # 环境配置
├── Dockerfile              # Docker构建文件
├── docker-compose.yml      # Docker Compose配置
├── requirements.txt        # 依赖列表
├── README.md               # 项目说明
├── 一键训练.bat             # Windows一键训练
└── 一键预测.bat             # Windows一键预测
```

## ❓ 疑难解答

### Docker相关问题
- **Docker未启动**: 确保Docker Desktop正在运行
- **WSL2问题**: Windows用户可能需要确保WSL2已安装且工作正常
- **端口占用**: 如遇端口冲突，可尝试修改docker-compose.yml中的端口映射

### 内存不足
- 运行`scripts/setup.bat`（Windows）或`scripts/setup.sh`（Linux）选择更低内存配置
- 手动编辑`.env`文件，减少`MAX_SEQ_LEN`和`MAX_SAMPLES`
- 增加`GRAD_ACCUM`参数值

### 训练缓慢
- CPU训练本身较慢，耐心等待
- 考虑减少`MAX_SAMPLES`参数
- 增加`NUM_THREADS`参数（如果您的CPU有多个核心）

### 模型下载失败
- 检查网络连接
- 修改`HF_ENDPOINT`环境变量为其他镜像站点
- 手动下载模型并放入缓存目录

### 量化失败
- 尝试重启训练
- 系统会自动回退到非量化模式（会占用更多内存）
- 尝试更新`bitsandbytes`和`transformers`版本

### Windows特有问题
- **路径过长**: Windows有时会遇到路径长度限制，可以将项目放在更短的路径下
- **权限问题**: 可能需要以管理员身份运行批处理脚本
- **中文乱码**: 确保脚本以ANSI或UTF-8(with BOM)编码保存

## 🔄 更新日志

### v1.1.0
- 添加Windows环境一键脚本支持
- 改进错误处理与用户提示
- 增强脚本健壮性

### v1.0.0
- 初始版本发布
- 支持ChatGLM2-6B模型训练
- 提供4bit和8bit量化支持
- 内存监控和性能追踪功能
- Docker容器化部署

## 📄 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。

## 🙏 致谢

- [THUDM/ChatGLM](https://github.com/THUDM/ChatGLM)：提供基础模型
- [HuggingFace/PEFT](https://github.com/huggingface/peft)：提供LoRA实现
- [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)：提供量化支持