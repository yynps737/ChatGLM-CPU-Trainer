"""
simple_train.py 中 load_model_and_tokenizer 函数的优化版本
"""

def load_model_and_tokenizer(args):
    """加载模型和分词器 - 针对低内存环境优化"""
    logger.info(f"加载模型: {args.model_name_or_path}")

    # 在加载模型之前先清理内存
    import gc
    gc.collect()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    # 确保分词器有正确的特殊标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检查可用内存，并提供警告
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    model_size_estimate = 6.0  # 假设模型大约为6GB (ChatGLM2-6B非量化)
    if "chatglm3" in args.model_name_or_path.lower():
        model_size_estimate = 12.0  # ChatGLM3-6B更大

    # 如果是ChatGLM3但内存少于8GB，自动切换到ChatGLM2
    if "chatglm3" in args.model_name_or_path.lower() and available_memory < 8.0 and args.quantization != "4bit":
        logger.warning(f"内存不足 ({available_memory:.1f}GB) 加载 ChatGLM3-6B 模型!")
        logger.warning(f"推荐: 1) 使用4bit量化 2) 切换到ChatGLM2-6B 3) 增加系统内存")

    # 加载模型的参数
    load_in_8bit = args.quantization == "8bit"
    load_in_4bit = args.quantization == "4bit"

    model_kwargs = {
        "trust_remote_code": True,
    }

    # 尝试启用量化
    try:
        if load_in_8bit or load_in_4bit:
            try:
                import bitsandbytes as bnb
                if load_in_8bit:
                    logger.info("使用8位量化加载模型")
                    model_kwargs["load_in_8bit"] = True
                else:
                    logger.info("使用4位量化加载模型 (超低内存模式)")
                    model_kwargs["load_in_4bit"] = True
                    # 启用Double量化以进一步减少内存使用
                    model_kwargs["bnb_4bit_use_double_quant"] = True
                    model_kwargs["bnb_4bit_quant_type"] = "nf4"
                    model_kwargs["bnb_4bit_compute_dtype"] = torch.float32
            except ImportError:
                logger.warning(f"bitsandbytes导入失败，尝试替代方法")
                # 尝试自动解决bitsandbytes问题
                if "Windows" in platform.system():
                    try:
                        logger.warning("尝试自动安装Windows兼容的bitsandbytes...")
                        # 尝试卸载现有版本
                        os.system("pip uninstall -y bitsandbytes-windows")
                        # 尝试安装Windows兼容版本
                        os.system("pip install bitsandbytes-windows")
                        import bitsandbytes as bnb
                        logger.info("成功安装bitsandbytes-windows")
                        if load_in_4bit:
                            model_kwargs["load_in_4bit"] = True
                        else:
                            model_kwargs["load_in_8bit"] = True
                    except:
                        logger.warning("自动安装失败，将使用非量化模式 (警告: 可能导致内存不足)")
                        model_kwargs["torch_dtype"] = torch.float32
                else:
                    model_kwargs["torch_dtype"] = torch.float32
        else:
            logger.info("使用32位精度加载模型 (警告: 内存使用量大)")
            model_kwargs["torch_dtype"] = torch.float32
    except Exception as e:
        logger.warning(f"量化设置失败: {e}")
        logger.warning("回退到32位精度 (警告: 内存使用量大)")
        model_kwargs["torch_dtype"] = torch.float32

    # 根据模型类型加载
    is_glm = is_chatglm_model(args.model_name_or_path)

    try:
        # 尝试逐步加载，以更好地处理低内存情况
        logger.info("开始加载模型...")

        # 确保清理任何已缓存的模型
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if is_glm:
            logger.info("检测到ChatGLM模型，使用专用加载方式")
            model = AutoModel.from_pretrained(
                args.model_name_or_path,
                **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                **model_kwargs
            )

        logger.info("模型加载成功!")

    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "DefaultCPUAllocator: can't allocate memory" in str(e):
            logger.error(f"内存不足错误: {e}")
            logger.error("请尝试: 1) 减少序列长度和样本数 2) 使用4bit量化 3) 使用更小的模型")
            raise RuntimeError("内存不足，无法加载模型。请参考上面的建议。")
        elif "bitsandbytes" in str(e):
            logger.error(f"BitsAndBytes错误: {e}")
            logger.error("尝试非量化模式...")
            # 尝试非量化模式
            try:
                if is_glm:
                    model = AutoModel.from_pretrained(
                        args.model_name_or_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float32
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float32
                    )
                logger.info("使用非量化模式成功加载模型")
            except Exception as e2:
                logger.error(f"非量化模式加载失败: {e2}")
                raise
        else:
            logger.error(f"加载模型时出错: {e}")
            raise

    return model, tokenizer