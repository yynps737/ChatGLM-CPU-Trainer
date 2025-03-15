# ChatGLM-CPU-Trainer

一个专为CPU环境优化的ChatGLM模型训练工具，使用LoRA技术在低资源环境下微调ChatGLM模型。

## 项目简介

ChatGLM-CPU-Trainer是一个为资源受限环境设计的模型训练工具，通过内存优化、量化技术和LoRA参数高效微调，使普通PC电脑也能进行大语言模型训练。项目特点：

- **低资源需求**：支持在8GB内存的普通电脑上训练
- **量化支持**：提供4-bit和8-bit量化选项，大幅降低内存占用
- **LoRA微调**：使用参数高效微调技术，只训练少量参数
- **Docker封装**：简化环境配置，确保兼容性
- **内存监控**：实时监控内存使用，防止OOM错误
- **简单界面**：提供一键式批处理脚本，无需编写命令

## 系统要求

- Windows操作系统（支持Docker）
- Docker Desktop已安装并运行
- 最低8GB内存（推荐16GB或以上）
- 20GB可用磁盘空间
- 网络连接（用于下载模型）

## 快速开始

### 一键安装与配置

1. 运行`setup.bat`进行系统配置检测和初始化
   ```
   setup.bat
   ```

2. 构建Docker镜像
   ```
   docker build -t chatglm-cpu-trainer .
   ```

### 训练模型

1. 准备训练数据，将文本放入`data/input/dataset.txt`文件中
   - 每行一个示例
   - 支持txt、csv、json、jsonl格式

2. 运行训练脚本
   ```
   一键训练.bat
   ```
   或使用docker-compose命令定制训练参数
   ```
   docker-compose run train
   ```

### 使用模型推理

训练完成后，使用以下命令测试模型：

```
一键预测.bat
```

或自定义提示词：

```
docker-compose run -e "PROMPT=你的提示词" predict
```

## 高级配置

### 内存配置

系统会根据检测到的内存自动选择对应配置：

| 内存配置 | 系统内存 | 量化方式 | 序列长度 | 样本数 | 批大小 |
|----------|---------|---------|---------|--------|--------|
| 4GB      | <6GB    | 4-bit   | 32      | 30     | 1      |
| 8GB      | <12GB   | 4-bit   | 64      | 200    | 1      |
| 16GB     | <24GB   | 8-bit   | 128     | 800    | 2      |
| 32GB     | ≥24GB   | 无量化   | 256     | 2000   | 4      |

可以在`.env`文件中手动调整这些参数。

### 量化选项

- **4-bit量化**：内存占用最小，约为原始模型的1/8
- **8-bit量化**：内存占用适中，约为原始模型的1/4
- **无量化**：内存占用最大，但精度最高

### 自定义训练参数

编辑`docker-compose.yml`文件或使用环境变量覆盖默认设置：

```
MAX_SAMPLES=50 docker-compose run train
```

## 项目结构

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
├── scripts/                  # 批处理脚本
├── .dockerignore             # Docker忽略文件
├── .env                      # 环境变量配置
├── Dockerfile                # Docker配置文件
├── docker-compose.yml        # Docker Compose配置
├── requirements.txt          # Python依赖
├── setup.bat                 # 环境配置脚本
├── 一键训练.bat               # 快速训练脚本
└── 一键预测.bat               # 快速预测脚本
```

## 常见问题

### 模型下载速度慢

如果遇到模型下载缓慢的问题，可以尝试：
1. 在`.env`文件中修改镜像站：`HF_ENDPOINT=https://hf-mirror.com`
2. 手动下载模型并放入缓存目录：`~/.cache/huggingface`

### 内存不足错误

如果训练过程中遇到内存不足错误：
1. 减少`MAX_SAMPLES`或`MAX_SEQ_LEN`参数
2. 使用更高级别的量化（从无量化切换到8-bit或4-bit）
3. 增加梯度累积步数`GRAD_ACCUM`
4. 确保关闭其他内存占用大的应用

### Docker相关问题

如果遇到Docker相关问题：
1. 确保Docker Desktop正在运行
2. 尝试重启Docker服务
3. 检查Docker资源限制是否合理

## 许可证

无

## 致谢

- ChatGLM模型由清华大学开发
- 使用了Hugging Face的Transformers、PEFT和bitsandbytes库
- 内存优化和量化技术借鉴了多个开源项目的最佳实践