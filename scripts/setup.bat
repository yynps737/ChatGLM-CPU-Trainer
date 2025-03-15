@echo off
chcp 65001
echo ===== 创建新训练环境 =====

:: 创建必要目录
mkdir data 2>nul
mkdir output 2>nul

:: 创建一个非常小的样本数据集
echo 创建样本数据集...
(
echo {"text": "这是一条测试数据，用于训练ChatGLM模型。"}
echo {"text": "人工智能是研究如何使计算机模拟人类智能的一门科学。"}
echo {"text": "深度学习是机器学习的一个分支，它基于人工神经网络进行学习。"}
) > data\sample.jsonl

:: 安装最小依赖集
echo 安装核心依赖...
pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.30.2 peft==0.4.0 datasets==2.13.0 accelerate==0.21.0

echo.
echo ===== 训练环境创建完成 =====
echo.
pause