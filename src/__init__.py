# ChatGLM-CPU-Trainer 主模块
from .models import (
    load_base_model,
    load_peft_model,
    create_peft_config,
    add_lora_to_model,
    save_model_and_tokenizer,
    is_chatglm_model
)

from .train import (
    load_training_dataset,
    prepare_dataset_for_training,
    prepare_instruction_dataset,
    create_chat_prompt,
    create_trainer,
    create_training_args,
    get_last_checkpoint
)

from .utils import (
    get_system_info,
    optimize_memory_usage,
    generate_optimized_ds_config,
    auto_detect_batch_size
)