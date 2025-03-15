.PHONY: help setup build train predict predict-custom batch-predict clean test debug monitor

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
	@echo "make batch-predict FILE=path/to/prompts.txt  - 批量预测多个提示"
	@echo "make clean        - 清理生成的文件和Docker资源"
	@echo "make test         - 运行基本测试确保环境正常"
	@echo "make debug        - 启动交互式调试会话"
	@echo "make monitor      - 监控训练进程内存和CPU使用情况"
	@echo ""
	@echo "高级用法:"
	@echo "make train MAX_SAMPLES=50  - 使用自定义参数值覆盖.env配置"
	@echo "make train OPTIMIZER=adafactor LEARNING_RATE=1e-4  - 自定义优化器和学习率"
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
	@if [ -f data/output/prediction.txt ]; then \
		cat data/output/prediction.txt; \
	else \
		echo "预测输出文件未找到"; \
	fi

# 自定义提示词预测
predict-custom: setup
	@[ -z "$(PROMPT)" ] && echo "错误: 请提供PROMPT参数" && exit 1 || true
	@echo "使用自定义提示进行预测: \"$(PROMPT)\""
	docker-compose run --rm -e PROMPT="$(PROMPT)" predict
	@echo "预测完成!"
	@echo "结果已保存到 data/output/prediction.txt"
	@if [ -f data/output/prediction.txt ]; then \
		cat data/output/prediction.txt; \
	else \
		echo "预测输出文件未找到"; \
	fi

# 批量预测
batch-predict: setup
	@if [ -z "$(FILE)" ]; then \
		echo "创建示例提示文件..."; \
		echo "请介绍一下人工智能的发展历史。" > data/input/prompts.txt; \
		echo "什么是机器学习？它有哪些应用场景？" >> data/input/prompts.txt; \
		echo "深度学习与传统机器学习有什么区别？" >> data/input/prompts.txt; \
		echo "使用默认的示例提示文件 data/input/prompts.txt"; \
		INPUT_FILE="prompts.txt"; \
	else \
		echo "使用提示文件: $(FILE)"; \
		cp "$(FILE)" data/input/custom_prompts.txt; \
		INPUT_FILE="custom_prompts.txt"; \
	fi; \
	docker-compose run --rm -e INPUT_FILE="$$INPUT_FILE" batch-predict \
		--input_file "/app/data/input/$$INPUT_FILE" \
		--output_file "/app/data/output/batch_results.json" \
		$(filter-out $@,$(MAKEOVERRIDES) $(MAKECMDGOALS))
	@echo "批量预测完成!"
	@echo "结果已保存到 data/output/batch_results.json 和 data/output/batch_results_summary.txt"
	@if [ -f data/output/batch_results_summary.txt ]; then \
		echo "生成摘要:"; \
		head -n 20 data/output/batch_results_summary.txt; \
		echo "... (更多内容请查看完整文件)"; \
	else \
		echo "摘要文件未找到"; \
	fi

# 测试环境
test: setup
	@echo "正在测试环境..."
	@echo "1. 检查Docker状态..."
	@docker info > /dev/null 2>&1 || (echo "Docker未运行!" && exit 1)
	@echo "2. 检查目录结构..."
	@[ -d "data/input" ] && [ -d "data/output" ] && [ -d "models" ] || (echo "目录结构不完整!" && exit 1)
	@echo "3. 验证模型环境..."
	docker-compose run --rm --entrypoint python -T predict -c "import torch; import transformers; print(f'PyTorch版本: {torch.__version__}'); print(f'Transformers版本: {transformers.__version__}')"
	@echo "4. 测试数据集访问..."
	@[ -f "data/input/dataset.txt" ] || (echo "数据集文件不存在!" && exit 1)
	@echo "环境测试完成，一切正常!"

# 交互式调试会话
debug: setup
	@echo "启动调试会话 (Python交互式Shell)..."
	docker-compose run --rm --entrypoint python -it predict

# 监控训练进程
monitor:
	@echo "监控训练进程..."
	@if [ -z "$$(docker ps | grep chatglm-train)" ]; then \
		echo "没有运行中的训练进程!"; \
		exit 1; \
	fi
	@echo "每5秒刷新一次状态 (按Ctrl+C退出)..."
	@for i in {1..100}; do \
		clear; \
		echo "===== ChatGLM-CPU-Trainer 监控 ====="; \
		echo "时间: $$(date)"; \
		echo ""; \
		echo "容器状态:"; \
		docker stats --no-stream chatglm-train; \
		echo ""; \
		echo "训练日志 (最后10行):"; \
		tail -n 10 data/output/train.log; \
		echo ""; \
		echo "内存使用:"; \
		tail -n 5 data/output/memory_usage.csv; \
		sleep 5; \
	done

# 清理生成的文件和Docker资源
clean:
	@echo "正在清理资源..."
	docker-compose down
	docker rmi chatglm-cpu-trainer || true
	@echo "是否要删除所有输出和模型文件? [y/N] "
	@read -r confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "删除输出文件..."; \
		rm -rf data/output/*; \
		echo "删除模型文件..."; \
		rm -rf models/*; \
		echo "所有文件已删除!"; \
	else \
		echo "保留输出和模型文件"; \
	fi
	@echo "清理完成!"

# 允许传递任意参数
%:
	@: