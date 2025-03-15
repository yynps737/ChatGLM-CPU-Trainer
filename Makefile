.PHONY: help setup build train predict predict-custom clean

SHELL := /bin/bash

# 默认目标
help:
	@echo "ChatGLM-CPU-Trainer 命令帮助"
	@echo "-----------------------------"
	@echo "make setup        - 自动检测系统并配置环境参数"
	@echo "make build        - 构建Docker镜像"
	@echo "make train        - 开始训练模型"
	@echo "make predict      - 使用默认提示进行预测"
	@echo "make predict-custom PROMPT=\"您的自定义提示\"  - 使用自定义提示进行预测"
	@echo "make clean        - 清理生成的文件和Docker资源"
	@echo ""
	@echo "高级用法:"
	@echo "make train MAX_SAMPLES=50  - 使用自定义参数值覆盖.env配置"
	@echo ""

# 构建Docker镜像
build:
	@echo "正在构建Docker镜像..."
	docker-compose build
	@echo "Docker镜像构建完成!"

# 自动配置环境
setup: build
	@echo "正在检测系统并配置环境..."
	@mkdir -p data/input data/output models
	@# 如果示例数据集不存在，则创建一个空的示例数据集文件
	@if [ ! -f "data/input/dataset.txt" ]; then \
		echo "创建示例数据集文件..."; \
		cp -n data/input/dataset.txt data/input/dataset.txt 2>/dev/null || \
		echo "人工智能是计算机科学的一个重要分支，致力于研发能够像人类一样思考和学习的智能机器。" > data/input/dataset.txt; \
	fi
	docker-compose run --rm setup
	@if [ -f "data/.env" ]; then \
		cp data/.env .env; \
		echo "配置完成! 环境配置已保存到.env文件"; \
	else \
		echo "配置过程出错"; \
		exit 1; \
	fi
	@echo "文件目录准备就绪。"
	@echo "您可以将训练数据放入data/input/dataset.txt文件中。"

# 训练模型
train: setup
	@echo "开始训练模型..."
	docker-compose run --rm train $(filter-out $@,$(MAKEOVERRIDES) $(MAKECMDGOALS))
	@echo "训练完成!"

# 默认提示词预测
predict: setup
	@echo "开始预测..."
	docker-compose run --rm predict $(filter-out $@,$(MAKEOVERRIDES) $(MAKECMDGOALS))
	@echo "预测完成!"
	@echo "结果已保存到 data/output/prediction.txt"
	@test -f data/output/prediction.txt && cat data/output/prediction.txt || echo "预测输出文件未找到"

# 自定义提示词预测
predict-custom: setup
	@[ -z "$(PROMPT)" ] && echo "错误: 请提供PROMPT参数" && exit 1 || true
	@echo "使用自定义提示进行预测: \"$(PROMPT)\""
	docker-compose run --rm -e PROMPT="$(PROMPT)" predict
	@echo "预测完成!"
	@echo "结果已保存到 data/output/prediction.txt"
	@test -f data/output/prediction.txt && cat data/output/prediction.txt || echo "预测输出文件未找到"

# 清理生成的文件和Docker资源
clean:
	@echo "正在清理资源..."
	docker-compose down
	docker rmi chatglm-cpu-trainer || true
	@echo "清理完成!"

# 允许传递任意参数
%:
	@: