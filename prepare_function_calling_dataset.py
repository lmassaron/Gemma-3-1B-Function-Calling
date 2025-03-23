"""
Script for preprocessing the Hermes function calling dataset.
Transforms conversation data format and splits into train/test sets.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Any

import random
from datasets import load_dataset, Dataset


def preprocess_dataset(
    dataset: Dataset,
    seed: int = 42,
    system_message_addition: str = "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n",
) -> Tuple[Dataset, Dataset]:
    """Preprocess conversation dataset by transforming message format and splitting into train/test"""

    # Set random seed for reproducibility
    random.seed(seed)
    print(f"Using random seed: {seed}")

    # Extract only the 'conversations' feature
    if "conversations" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'conversations' column")

    dataset = dataset.select_columns(["conversations"])
    print(f"Processing dataset with {len(dataset)} examples")

    def transform_conversation(
        example: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, str]]]:
        """Transform conversation format from HF dataset to standardized format."""
        conversations = example["conversations"]

        # Rename 'from' to 'role' and 'value' to 'content'
        # Also rename 'gpt' role to 'model'
        messages = []
        for msg in conversations:
            role = msg["from"]
            if role == "gpt":
                role = "model"

            messages.append({"role": role, "content": msg["value"]})

        # Handle system message if it's the first message
        if messages and messages[0]["role"] == "system" and len(messages) > 1:
            system_message_content = messages[0]["content"]
            # Merge system content with the first user message
            messages[1]["content"] = (
                system_message_content
                + system_message_addition
                + messages[1]["content"]
            )
            # Remove the system message
            messages.pop(0)

        # Return the transformed conversations with the same key name
        return {"conversations": messages}

    # Apply the transformation
    print("Transforming conversation format...")
    transformed_dataset = dataset.map(transform_conversation)

    # Split into two datasets
    print("Splitting into train/test sets (80/20)...")
    train_test_split = transformed_dataset.train_test_split(test_size=0.2, seed=seed)

    print(f"Train set: {len(train_test_split['train'])} examples")
    print(f"Test set: {len(train_test_split['test'])} examples")

    return train_test_split["train"], train_test_split["test"]


def main():
    """Main function to process dataset and save results."""
    dataset_name = "NousResearch/hermes-function-calling-v1"
    dataset_config = "glaive_func_calling"
    output_dir = "./processed_data"
    seed = 42

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_name} (config: {dataset_config})")
    dataset = load_dataset(dataset_name, dataset_config, split="train")

    # Process the dataset
    train_dataset, test_dataset = preprocess_dataset(dataset, seed=seed)

    # Save datasets to disk
    train_path = os.path.join(output_dir, "function_calling_train")
    test_path = os.path.join(output_dir, "function_calling_test")

    print(f"Saving train dataset to {train_path}")
    train_dataset.save_to_disk(train_path)

    print(f"Saving test dataset to {test_path}")
    test_dataset.save_to_disk(test_path)

    print("Processing complete!")


if __name__ == "__main__":
    main()
