# function_calling_gemma3

This project demonstrates how to fine-tune the `google/gemma-3-1b-it` and `google/gemma-3-270m-it`models for function calling tasks. The process involves preparing the `NousResearch/hermes-function-calling-v1` dataset, fine-tuning the model using Low-Rank Adaptation (LoRA), and evaluating its performance on a test set.

You can find a detailed description of the project by reading my Medium article:
* [Fine-Tuning Gemma 3 1B for Function Calling: A Step-by-Step Guide](https://medium.com/@lucamassaron/fine-tuning-gemma-3-1b-for-function-calling-a-step-by-step-guide-66a613352f99)

![Function calling as a toolbox](./toolbox.png)

The project is structured into three main Python scripts:

*   `prepare_function_calling_dataset.py`: Downloads and preprocesses the dataset, splitting it into training and testing sets.
*   `function_calling_train.py`: Fine-tunes the Gemma model using the preprocessed training data and LoRA.
*   `evaluation.py`: Evaluates the fine-tuned model's performance on function calling and helpful exchanges.

## About the Project

This project provides a complete workflow for fine-tuning a large language model for function calling. By following the scripts, you can reproduce the results and adapt the process for your own function calling tasks. The use of LoRA allows for efficient fine-tuning, and the evaluation script provides a clear measure of the model's performance improvement.

Baseline, with unquantized Gemma 3-1b-it:

* Accuracy in function calling: ~0.27905
* Match in helpful exchange: ~0.28897

After finetuning:
* Accuracy in function calling: ~0.90814
* Match in helpful exchange: ~0.77873

Baseline, with unquantized Gemma 3-270m-it:

* Accuracy in function calling: ~0.12805
* Match in helpful exchange: ~0.48171

After finetuning:
* Accuracy in function calling: ~0.90426
* Match in helpful exchange: ~0.67602


## Getting Started

To get started with this project, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/lmassaron/Gemma-3-1B-Function-Calling.git
    cd Gemma-3-1B-Function-Calling
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the scripts:**

    *   First, prepare the dataset:

        ```bash
        python prepare_function_calling_dataset.py
        ```

    *   Then, fine-tune the model:

        ```bash
        python function_calling_train.py
        ```

    *   Finally, evaluate the model:

        ```bash
        python evaluation.py
        ```

## Model and Training

The base model for this project is `google/gemma-3-1b-it` or `google/gemma-3-270m-it`. The model is fine-tuned using Low-Rank Adaptation (LoRA), which is a parameter-efficient fine-tuning technique. The LoRA configuration can be found in the `function_calling_train.py` script.

The training process uses the `trl` library and the `SFTTrainer` to perform supervised fine-tuning. The training arguments, such as the number of epochs, batch size, and learning rate, are also defined in the `function_calling_train.py` script.
