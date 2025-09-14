"""Evaluating a fine-tuned language model for function calling"""

from collections import Counter
import numpy as np
import torch
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from peft import PeftModel, PeftConfig
from tqdm import tqdm


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


def determine_compute_dtype():
    """Determine the appropriate compute dtype based on CUDA capabilities"""
    try:
        # Check for NVIDIA Ampere architecture (8.0) or newer
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            # Use bfloat16 for better training stability on newer GPUs
            compute_dtype = torch.bfloat16
        else:
            # Fall back to float16 for older GPUs
            compute_dtype = torch.float16
    except (RuntimeError, AttributeError, IndexError) as e:
        # Handle exceptions if CUDA is not available and you are using CPU or MPS
        print(f"Error determining compute dtype: {e}")  # Log the exception
        compute_dtype = torch.float16

    return compute_dtype


def compute_matching_percentage(list1, list2):
    """Computes the percentage of matching elements between two lists."""
    if not list1 or not list2:
        return 0.0
    count1, count2 = Counter(list1), Counter(list2)
    matches = sum(min(count1[code], count2[code]) for code in count1 if code in count2)
    return matches / len(list2)


def find_longest_common_sequence_length(list1, list2):
    """Finds the length of the longest common contiguous sequence between two lists."""
    if not list1 or not list2:
        return 0
    m, n = len(list1), len(list2)
    prev_row = [0] * (n + 1)
    current_row = [0] * (n + 1)
    max_length = 0
    for i in range(1, m + 1):
        prev_row, current_row = current_row, prev_row
        for j in range(1, n + 1):
            if list1[i - 1] == list2[j - 1]:
                current_row[j] = prev_row[j - 1] + 1
                max_length = max(max_length, current_row[j])
            else:
                current_row[j] = 0
    return max_length


def extract_last_model_turn(raw_output):
    """Extracts the content from the last model turn in a raw generated string"""
    start_tag = "<start_of_turn>"
    end_tag = "<end_of_turn>"

    last_start_pos = raw_output.rfind(start_tag)

    if last_start_pos == -1:
        return raw_output.strip()

    last_end_pos = raw_output.find(end_tag, last_start_pos)

    content_start_pos = last_start_pos + len(start_tag)

    if last_end_pos != -1:
        content = raw_output[content_start_pos:last_end_pos]
    else:
        content = raw_output[content_start_pos:]

    first_newline = content.find("\n")
    if first_newline != -1:
        content = content[first_newline + 1 :]

    return content.replace("<eos>", "").replace("<pad>", "").strip()


def generate_from_model_batch(batch_conversations, model, tokenizer):
    # Ensure proper chat template application, including generation prompt
    prompts = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,  # Add generation prompt for chat models
        )
        for conv in batch_conversations
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
    ).to(model.device)  # Use model.device if 'device' is not globally defined

    prompt_actual_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.95,
        temperature=0.01,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,  #  Explicitly set pad_token_id
    )

    generated_decoded = []
    for i, output_tensor in enumerate(outputs):
        # Use the accurate prompt_actual_lengths for slicing
        generated_only_ids = output_tensor[prompt_actual_lengths[i] :]

        # Decode without skipping special tokens here.
        decoded_text = tokenizer.decode(generated_only_ids, skip_special_tokens=False)
        clean_output = extract_last_model_turn(decoded_text)
        generated_decoded.append(
            clean_output.strip()
        )  # Strip leading/trailing whitespace

    return generated_decoded


def evaluate_function_calling(dataset, model, tokenizer, batch_size=8):
    """Evaluates a model on a function-calling dataset"""
    # Suppress the warning by setting the pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    test_examples = len(dataset)
    tooling = []
    being_useful = []
    queries = []
    answers = []

    prompts_list = []
    returned_outputs_list = []
    expected_outputs_list = []
    tool_call_flag_list = []

    for i in range(test_examples):
        conversations = []
        for item in dataset[i]["conversations"]:
            if item["role"] != "model":
                conversations.append(item)
            if item["role"] == "model":
                queries.append(conversations[:])
                answers.append(item["content"])
                conversations.append(item)

    batches = [queries[i : i + batch_size] for i in range(0, len(queries), batch_size)]
    generated_outputs = []
    for batch in tqdm(batches):
        # Assuming generate_from_model_batch is defined elsewhere
        generated_outputs.extend(generate_from_model_batch(batch, model, tokenizer))

    for i, (ground_truth, generated) in enumerate(zip(answers, generated_outputs)):
        ground_truth_tokens = tokenizer(ground_truth)["input_ids"]
        generated_tokens = tokenizer(generated)["input_ids"]

        is_tool_call = "<tool_call>" in ground_truth
        tool_call_flag_list.append(is_tool_call)
        prompts_list.append(
            tokenizer.decode(tokenizer(str(queries[i]))["input_ids"])
        )  # This remains the same
        returned_outputs_list.append(generated)
        expected_outputs_list.append(ground_truth)

        if is_tool_call:
            seq = find_longest_common_sequence_length(
                ground_truth_tokens, generated_tokens
            )
            matches = (
                seq / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            )
            tooling.append(matches)
        else:
            matches = compute_matching_percentage(ground_truth_tokens, generated_tokens)
            being_useful.append(matches)

    torch.cuda.empty_cache()

    print(
        f"\nAccuracy in function calling: {np.mean(tooling) if tooling else 0.0:0.5f}"
    )
    print(
        f"Match in helpful exchange: {np.mean(being_useful) if being_useful else 0.0:0.5f}"
    )

    results_df = pd.DataFrame(
        {
            "prompt": prompts_list,
            "returned_output": returned_outputs_list,
            "expected_output": expected_outputs_list,
            "tool_call": tool_call_flag_list,
        }
    )

    return results_df


if __name__ == "__main__":
    BATCH_SIZE = 24
    torch.manual_seed(42)  # For reproducibility
    device = define_device()
    compute_dtype = determine_compute_dtype()

    peft_model_id = "LoRA_gemma-3-1b-it-function_calling"
    peftconfig = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        peftconfig.base_model_name_or_path,
        attn_implementation="eager",
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, peft_model_id)
    model = model.to(torch.bfloat16)
    model = model.eval()

    dataset_name = "processed_data/function_calling_test"
    dataset = load_from_disk(dataset_name)

    results_dataframe = evaluate_function_calling(
        dataset, model, tokenizer, batch_size=BATCH_SIZE
    )

    results_dataframe.to_csv(f"{peft_model_id}_eval.csv", index=False)
