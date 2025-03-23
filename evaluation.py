"""Evaluating a fine-tuned language model for function calling"""

from collections import Counter
import numpy as np
import torch

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


def generate_from_model(conversations):
    # Tokenize and move input to the appropriate device
    prompt = tokenizer.apply_chat_template(conversations, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    # Generate output from the model
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.95,
        temperature=0.01,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Tokenize the prompt and output separately
    output_tokens = outputs[0]  # Assuming outputs is a list of token IDs

    # Get the correct token length of the prompt
    prompt_length = inputs["input_ids"].shape[1]  # Number of tokens in the prompt

    # Decode only the generated output
    generated_decoded = tokenizer.decode(
        output_tokens[prompt_length:], skip_special_tokens=False
    )

    return generated_decoded


if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducibility
    device = define_device()
    compute_dtype = determine_compute_dtype()

    peft_model_id = "LoRA_gemma-3-1B-it-function_calling"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
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

    test_examples = len(dataset)
    tooling = []
    being_useful = []

    for i in tqdm(range(test_examples)):
        answers = []
        conversations = []
        for item in dataset[i]["conversations"]:
            if item["role"] != "model":
                conversations.append(item)
            if item["role"] == "model":
                generated = generate_from_model(
                    conversations
                )  # only using generated output
                answers.append(
                    [
                        conversations,
                        item["content"],
                        generated.strip(),
                    ]  # save conversations for debugging
                )
                conversations.append(item)

        for _, ground_truth, generated in answers:
            ground_truth_tokens = tokenizer(ground_truth)["input_ids"]
            generated_tokens = tokenizer(generated)["input_ids"]

            # Evaluate function calling accuracy if tool call is present
            if "<tool_call>" in ground_truth:
                seq = find_longest_common_sequence_length(
                    ground_truth_tokens, generated_tokens
                )
                matches = seq / len(ground_truth_tokens)
                tooling.append(matches)
            else:
                matches = compute_matching_percentage(
                    ground_truth_tokens, generated_tokens
                )
                being_useful.append(matches)

    print(f"Accuracy in function calling: {np.mean(tooling):0.5f}")
    print(f"Match in helpful exchange: {np.mean(being_useful):0.5f}")
