FROM python:3.8-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    HF_ENDPOINT=https://hf-mirror.com \
    CUDA_VISIBLE_DEVICES=""

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 升级pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装基础依赖
RUN pip install --no-cache-dir \
    torch==1.13.1 \
    numpy==1.21.6 \
    pandas==1.3.5 \
    scipy==1.7.3 \
    tqdm==4.64.1

# 安装Hugging Face依赖 - 确保版本兼容
RUN pip install --no-cache-dir \
    transformers==4.30.2 \
    datasets==2.12.0 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3

# 安装PEFT和加速相关依赖
RUN pip install --no-cache-dir \
    peft==0.4.0 \
    accelerate==0.20.3

# 安装CPU版本的bitsandbytes - 确保与PEFT兼容
RUN pip install --no-cache-dir bitsandbytes==0.37.0

# 复制应用代码
COPY ./app /app/app/
COPY ./scripts /app/scripts/
COPY ./requirements.txt /app/requirements.txt
COPY ./README.md /app/README.md

# 创建必要的目录
RUN mkdir -p data/input data/output models

# 添加Python模块路径
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 设置卷
VOLUME ["/app/data", "/app/models"]

# 默认命令
CMD ["python", "-c", "import sys; print('请使用特定命令启动容器')"]