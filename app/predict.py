"""
ChatGLM 预测脚本 - 用于验证训练后的模型 (Docker优化版)
"""

import os
import argparse
import torch
import gc
import json
import time
from datetime import datetime
from tqdm import tqdm
from peft import PeftModel

# 导入工具函数
from app.utils import setup_logging, load_model_and_tokenizer
from app.memory_monitor import MemoryMonitor

# 禁用CUDA，仅使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 设置日志
logger = setup_logging(log_file="/app/data/output/predict.log")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ChatGLM模型预测脚本 (Docker优化版)")

    parser.add_argument("--model_path", type=str, default="/app/models/chatglm-lora",
                        help="LoRA模型路径")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="基础模型路径")
    parser.add_argument("--prompt", type=str, default="请介绍一下人工智能的发展历史。",
                        help="测试提示")
    parser.add_argument("--input_file", type=str, default=None,
                        help="包含多个提示的输入文件，每行一个提示")
    parser.add_argument("--quantization", type=str, default="None",
                        choices=["4bit", "8bit", "None"],
                        help="模型量化类型")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="生成的最大长度")
    parser.add_argument("--output_file", type=str, default="/app/data/output/prediction.txt",
                        help="输出文件路径")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度, 控制输出的随机性，越高越随机")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="核采样阈值，控制词汇选择的多样性")
    parser.add_argument("--top_k", type=int, default=40,
                        help="选择前k个最可能的词进行采样")
    parser.add_argument("--monitor_memory", action="store_true",
                        help="启用内存监控")
    parser.add_argument("--batch_mode", action="store_true",
                        help="批处理模式，对输入文件中的每个提示进行生成")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="为每个提示生成的回复数量")
    parser.add_argument("--verbose", action="store_true",
                        help="输出详细的预测过程信息")

    return parser.parse_args()


def load_model_for_inference(args):
    """加载基础模型和LoRA模型"""
    logger.info(f"加载基础模型: {args.base_model_path}")

    # 加载基础模型和分词器
    base_model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.quantization)

    # 加载LoRA模型
    logger.info(f"加载LoRA模型: {args.model_path}")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            args.model_path
        )
        logger.info("LoRA模型加载成功!")
    except Exception as e:
        logger.error(f"加载LoRA模型时出错: {e}")
        logger.warning("尝试直接使用基础模型...")
        model = base_model

    return model, tokenizer


def generate_response(model, tokenizer, prompt, args, verbose=False):
    """生成回复"""
    if verbose:
        logger.info(f"提示: {prompt}")

    start_time = time.time()

    try:
        # 尝试使用chat方法生成回复
        response, history = model.chat(
            tokenizer,
            prompt,
            history=[],
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        method = "chat"
    except Exception as e:
        logger.error(f"使用chat方法生成回复时出错: {e}")

        # 尝试备用生成方法
        logger.warning("尝试使用备用生成方法...")
        try:
            # 先进行分词
            inputs = tokenizer(prompt, return_tensors="pt")

            # 生成回复
            gen_kwargs = {
                "max_length": args.max_length,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "do_sample": True,
                "num_return_sequences": args.num_return_sequences,
            }

            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                **gen_kwargs
            )

            # 解码生成的token
            if args.num_return_sequences > 1:
                responses = []
                for i in range(args.num_return_sequences):
                    response_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                    # 移除输入提示部分
                    if response_text.startswith(prompt):
                        response_text = response_text[len(prompt):].strip()
                    responses.append(response_text)
                response = responses
            else:
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 移除输入提示部分
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
            method = "generate"
        except Exception as e2:
            logger.error(f"备用生成方法也失败: {e2}")
            raise

    end_time = time.time()
    generation_time = end_time - start_time

    if verbose:
        logger.info(f"生成回复成功! 使用方法: {method}, 耗时: {generation_time:.2f}秒")

    return response


def load_prompts_from_file(file_path):
    """从文件加载提示"""
    logger.info(f"从文件加载提示: {file_path}")
    prompts = []

    try:
        # 检测文件类型
        if file_path.endswith('.json') or file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                # 尝试解析为JSON或JSONL
                try:
                    # 作为单个JSON对象或数组加载
                    data = json.load(f)
                    if isinstance(data, list):
                        # JSON数组
                        for item in data:
                            if isinstance(item, str):
                                prompts.append(item)
                            elif isinstance(item, dict) and 'prompt' in item:
                                prompts.append(item['prompt'])
                    elif isinstance(data, dict) and 'prompts' in data:
                        # JSON对象带prompts字段
                        for prompt in data['prompts']:
                            if isinstance(prompt, str):
                                prompts.append(prompt)
                            elif isinstance(prompt, dict) and 'text' in prompt:
                                prompts.append(prompt['text'])
                except json.JSONDecodeError:
                    # 尝试作为JSONL处理
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            if isinstance(item, str):
                                prompts.append(item)
                            elif isinstance(item, dict) and 'prompt' in item:
                                prompts.append(item['prompt'])
                        except json.JSONDecodeError:
                            logger.warning(f"忽略无效的JSON行: {line[:50]}...")
        else:
            # 作为文本文件处理，每行一个提示
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prompts.append(line)
    except Exception as e:
        logger.error(f"加载提示文件出错: {e}")
        raise

    logger.info(f"从文件中加载了 {len(prompts)} 个提示")
    return prompts


def batch_generate(model, tokenizer, prompts, args):
    """批量生成回复"""
    logger.info(f"开始批量生成，共 {len(prompts)} 个提示")

    results = []

    for i, prompt in enumerate(tqdm(prompts, desc="生成进度")):
        try:
            response = generate_response(model, tokenizer, prompt, args)
            results.append({
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"生成第 {i+1} 个提示的回复时出错: {e}")
            results.append({
                "prompt": prompt,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

        # 每10个提示清理一次内存
        if (i + 1) % 10 == 0:
            gc.collect()

    return results


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置CPU环境变量
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "4")

    logger.info("=" * 50)
    logger.info("ChatGLM 模型预测脚本 (Docker优化版)")
    logger.info("=" * 50)
    logger.info(f"基础模型: {args.base_model_path}")
    logger.info(f"LoRA模型: {args.model_path}")
    logger.info(f"量化级别: {args.quantization}")
    logger.info(f"生成参数: 温度={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    logger.info("=" * 50)

    # 初始化内存监控
    memory_monitor = None
    if args.monitor_memory:
        from app.memory_monitor import MemoryMonitor
        logger.info("启用内存监控")
        memory_monitor = MemoryMonitor(logger=logger)
        memory_monitor.start_monitoring()

    try:
        # 加载模型和分词器
        model, tokenizer = load_model_for_inference(args)

        # 设置模型为评估模式
        model.eval()

        # 如果内存监控启用，打印模型加载后的内存使用
        if memory_monitor:
            memory_monitor.print_memory_info(detailed=True)

        # 确定是单个提示还是批处理模式
        if args.batch_mode or args.input_file:
            if not args.input_file:
                logger.error("批处理模式需要指定input_file参数")
                return

            # 加载提示
            prompts = load_prompts_from_file(args.input_file)
            if not prompts:
                logger.error("没有从文件中加载到有效提示")
                return

            # 批量生成
            logger.info("开始批量生成回复...")
            results = batch_generate(model, tokenizer, prompts, args)

            # 保存结果
            output_dir = os.path.dirname(args.output_file)
            os.makedirs(output_dir, exist_ok=True)

            # 确定输出文件格式
            if args.output_file.endswith('.json'):
                output_file = args.output_file
            else:
                output_file = os.path.splitext(args.output_file)[0] + '.json'

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"批量生成完成，结果已保存到: {output_file}")

            # 创建一个简单的文本摘要
            summary_file = os.path.splitext(args.output_file)[0] + '_summary.txt'
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"批量生成摘要 - {datetime.now().isoformat()}\n")
                f.write(f"共处理 {len(results)} 个提示\n\n")

                for i, result in enumerate(results):
                    f.write(f"[提示 {i+1}]\n{result['prompt']}\n\n")
                    if 'response' in result:
                        if isinstance(result['response'], list):
                            for j, resp in enumerate(result['response']):
                                f.write(f"[回复 {i+1}.{j+1}]\n{resp}\n\n")
                        else:
                            f.write(f"[回复 {i+1}]\n{result['response']}\n\n")
                    else:
                        f.write(f"[错误]\n{result.get('error', '未知错误')}\n\n")
                    f.write("-" * 50 + "\n\n")

            logger.info(f"生成摘要已保存到: {summary_file}")

        else:
            # 单个提示模式
            logger.info("开始生成回复...")
            response = generate_response(model, tokenizer, args.prompt, args, verbose=True)

            # 输出结果
            logger.info("生成完成!")
            print("\n" + "=" * 50)
            print("生成的回复:")
            print("-" * 50)
            if isinstance(response, list):
                for i, resp in enumerate(response):
                    print(f"回复 {i+1}:")
                    print(resp)
                    print("-" * 30)
            else:
                print(response)
            print("=" * 50)

            # 保存到文件
            output_dir = os.path.dirname(args.output_file)
            os.makedirs(output_dir, exist_ok=True)

            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(f"提示: {args.prompt}\n\n")
                f.write(f"回复:\n")
                if isinstance(response, list):
                    for i, resp in enumerate(response):
                        f.write(f"[回复 {i+1}]\n{resp}\n\n")
                else:
                    f.write(response)

            logger.info(f"结果已保存到: {args.output_file}")

    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 停止内存监控
        if memory_monitor:
            memory_monitor.stop_monitoring()

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()