# -FineTuning-with-PEFT-and-LoRa


## Overview
This repository demonstrates how to fine-tune large language models using Parameter-efficient Fine-tuning (PEFT) and Low-Rank Adaptation (LoRa) techniques. PEFT and LoRa offer a way to efficiently adapt large models like BLOOM without the need for extensive resources.

## Key Concepts
- **Parameter-efficient Fine-tuning (PEFT)**: PEFT is an approach that allows for the fine-tuning of large models with a small fraction of trainable parameters, significantly reducing the computational cost.
- **Low-Rank Adaptation (LoRa)**: LoRa is a technique for adapting pre-trained models by introducing low-rank matrices into the model's architecture, offering a balance between performance and efficiency.

## Installation
To set up the environment for fine-tuning, install the necessary libraries using the following commands:

```bash
!pip install -q bitsandbytes datasets accelerate loralib
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
```

## Setting up the Model
The script initializes a pre-trained BLOOM model and tokenizes the input data. It includes steps for freezing the original weights of the model and setting up LoRa adapters. The `print_trainable_parameters` function helps monitor the number of trainable parameters.

## Data Preparation
The script loads and preprocesses a dataset for training. It demonstrates how to merge columns and map the dataset to a tokenizer for language model training.

## Training
The training process is managed using Hugging Face's `Trainer` class. This section includes configuration for training arguments, such as batch size, learning rate, and the number of steps.

## Pushing to Hugging Face Hub
After training, the model can be pushed to the Hugging Face Hub using `model.push_to_hub`, allowing for easy sharing and collaboration.

## Loading Adapters from the Hub
This section illustrates how to load the trained LoRa model from the Hugging Face Hub for inference.

## Inference
Demonstrates the use of the fine-tuned model for generating predictions.

---

Note: This README provides a high-level overview. The actual code is available in the repository.
