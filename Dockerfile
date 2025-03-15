FROM python:3.10-slim-bullseye

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    HF_ENDPOINT=https://hf-mirror.com \
    CUDA_VISIBLE_DEVICES=""

# 安装系统依赖（一次性操作以减少层数）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 升级pip并安装依赖（一次性操作以减少层数）
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY ./app /app/app/
COPY ./README.md /app/README.md

# 创建必要的目录
RUN mkdir -p data/input data/output models

# 添加Python模块路径
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 设置卷
VOLUME ["/app/data", "/app/models"]

# 提供帮助信息的默认命令
CMD ["python", "-c", "print('ChatGLM-CPU-Trainer\\n\\n可用命令:\\n- python -m app.train --help\\n- python -m app.predict --help\\n\\n详细用法请参考README.md或使用docker-compose')"]