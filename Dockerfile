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

# 复制并安装依赖（一次性安装以确保兼容性）
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY ./app /app/app/
COPY ./scripts /app/scripts/
COPY ./README.md /app/README.md

# 创建必要的目录
RUN mkdir -p data/input data/output models

# 添加Python模块路径
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 设置卷
VOLUME ["/app/data", "/app/models"]

# 默认命令
CMD ["python", "-c", "import sys; print('请使用特定命令启动容器')"]