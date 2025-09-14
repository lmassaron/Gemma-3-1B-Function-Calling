"""Fine-tuning a Gemma 3-1B model for function calling using LoRA"""

from enum import Enum
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType


seed = 42
set_seed(seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenization para
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Put your HF Token here
# os.environ["HF_TOKEN"] = "hf_xxxxxxx"  # the token should have write access


def define_device():
    """Determine and return the optimal PyTorch device based on availability."""

    print(f"PyTorch version: {torch.__version__}", end=" -- ")

    # Check if MPS (Metal Performance Shaders) is available for macOS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("using MPS device on macOS")
        return torch.device("mps")

    # Check for CUDA availability
    detected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {detected_device}")
    return detected_device


def determine_compute_dtype(config):
    """Determine the appropriate compute dtype based on CUDA capabilities"""
    try:
        # Check for NVIDIA Ampere architecture (8.0) or newer
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            # Use bfloat16 for better training stability on newer GPUs
            compute_dtype = torch.bfloat16
            config.fp16 = False
            config.bf16 = True
        else:
            # Fall back to float16 for older GPUs
            compute_dtype = torch.float16
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                config.fp16 = False
            else:
                config.fp16 = True
            config.bf16 = False
    except (RuntimeError, AttributeError, IndexError) as e:
        # Handle exceptions if CUDA is not available and you are using CPU or MPS
        print(f"Error determining compute dtype: {e}")  # Log the exception
        compute_dtype = torch.float16
        config.fp16 = False
        config.bf16 = False

    return compute_dtype


def preprocess_and_filter(sample):
    """Preprocesses and filters a sample based on token length"""
    messages = sample["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer.encode(text, truncation=False)

    if len(tokens) <= config.training_arguments["max_length"]:
        return {"text": text}
    else:
        return None


class ChatmlSpecialTokens(str, Enum):
    """Enum class defining special tokens used in the ChatML format"""

    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call = "<tool_call>"
    eotool_call = "</tool_call>"
    tool_response = "<tool_response>"
    eotool_response = "</tool_response>"
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class Config:
    model_name = "google/gemma-3-1b-it"
    dataset_name = "processed_data/function_calling_train"
    output_dir = "gemma-3-1B-it-function_calling"  # The directory where the trained model checkpoints, logs, and other artifacts will be saved. It will also be the default name of the model when pushed to the hub if not redefined later.
    username = "lmassaron"
    lora_arguments = {
        "r": 16,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": [
            "embed_tokens",
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "o_proj",
            "lm_head",
        ],
    }
    training_arguments = {
        # Basic training configuration
        "num_train_epochs": 1,
        "max_steps": -1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_length": 4096,
        "packing": True,
        # Optimization settings
        "optim": "adamw_torch_fused",
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        # Memory optimization
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # Evaluation and saving
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        # Logging and output
        "logging_steps": 5,
        "report_to": "tensorboard",
        "logging_dir": "logs/runs",
        "overwrite_output_dir": True,
        # Model sharing
        "push_to_hub": True,
        "hub_private_repo": False,
    }
    fp16 = False
    bf16 = True
    batch_size = 24


if __name__ == "__main__":
    config = Config()

    # Determine optimal computation dtype based on GPU capability
    compute_dtype = determine_compute_dtype(config)
    print("compute dtype:", compute_dtype)

    # Select the best available device (CPU, CUDA, or MPS)
    device = define_device()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list(),
    )

    tokenizer.chat_template = tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}{% if message['role'] != 'system' %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=compute_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    dataset = (
        load_from_disk(config.dataset_name)
        .rename_column("conversations", "messages")
        .map(preprocess_and_filter, remove_columns="messages")
        .filter(lambda x: x is not None, keep_in_memory=True)
        .train_test_split(0.2)
    )

    peft_config = LoraConfig(
        **config.lora_arguments,
        task_type=TaskType.CAUSAL_LM,
    )

    training_arguments = SFTConfig(
        **config.training_arguments,
        output_dir=config.output_dir,
        fp16=config.fp16,
        bf16=config.bf16,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Saving LoRA weights and tokenizer
    trainer.model.save_pretrained(
        "LoRA_" + config.output_dir, save_embedding_layers=True
    )
    tokenizer.eos_token = "<eos>"
    tokenizer.save_pretrained("LoRA_" + config.output_dir)
