# Qwen 2.5 3B Fine-tuning for AI Research QA

This repository contains the code for fine-tuning the Qwen 2.5 3B model using a custom LoRA (Low-Rank Adaptation) approach. The project aims to enhance the model's performance on technical AI research literature and includes detailed documentation on data preprocessing, model adaptation, training, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
  - [Using Conda](#using-conda)
  - [Setting Up VS Code with Jupyter](#setting-up-vs-code-with-jupyter)
- [Dataset Organization](#dataset-organization)
- [Installation](#installation)
- [Running the Model](#running-the-model)
- [Troubleshooting](#troubleshooting)
- [Challenges Faced](#challenges-faced)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project fine-tunes the Qwen 2.5 3B model for AI research question-answering by using synthetic datasets and a parameter-efficient LoRA approach. It is designed for research and prototyping with provisions for scaling up and incorporating quantization and inference pipelines later.

## Features

- **Data Preprocessing:** Extract Q&A pairs from formatted text.
- **LoRA Integration:** Apply LoRA layers to target modules for efficient fine-tuning.
- **Custom Dataset:** Use a tailored PyTorch dataset for fine-tuning.
- **Training Pipeline:** Implements a training loop with detailed logging.
- **Model Saving:** Persist LoRA-specific weights for further use or quantization.

## Requirements

- Python 3.8+
- CUDA-enabled GPU (recommended for training acceleration)
- Conda (for environment management)
- VS Code with Jupyter extension

## Environment Setup

### Using Conda

We recommend creating a dedicated Conda environment to ensure all dependencies are managed properly. Follow these steps:

1. **Create a Conda Environment:**

   ```bash
   conda create -n qwen_finetune python=3.8
   ```

2. **Activate the Environment:**

   ```bash
   conda activate qwen_finetune
   ```

3. **Install Jupyter and Required Packages:**

   ```bash
   conda install jupyter ipykernel
   pip install notebook
   ```

4. Install CUDA Toolkit (if not already installed) and other dependencies as required by your system.
   You may need to follow the official NVIDIA installation guide for your specific OS.

### Setting Up VS Code with Jupyter

1. **Install VS Code Extensions:**
   - Open VS Code
   - Go to the Extensions view (Ctrl+Shift+X)
   - Search for and install:
     - "Python" extension by Microsoft
     - "Jupyter" extension by Microsoft

2. **Configure VS Code to Use Your Conda Environment:**
   - Open VS Code
   - Press Ctrl+Shift+P to open the Command Palette
   - Type "Python: Select Interpreter" and select it
   - Choose your `qwen_finetune` Conda environment from the list

3. **Create a New Jupyter Notebook:**
   - In VS Code, click on the "New File" button
   - Save the file with a `.ipynb` extension
   - VS Code will automatically open it as a Jupyter Notebook

4. **Select Kernel:**
   - Click on "Select Kernel" in the top-right corner of the notebook
   - Choose your `qwen_finetune` Conda environment

## Dataset Organization

Create a folder named `q3_dataset` in the project root. Place all your datasets (e.g., synthetic QA pairs, quotes datasets, research papers) in this folder. For example:

```bash
mkdir q3_dataset
# Place your dataset files inside the q3_dataset folder.
```

Ensure your code or data loader references this folder to load datasets.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/qwen_finetune.git
   cd qwen_finetune
   ```

2. **Install Python Dependencies:**

   Install all required packages using the provided requirements.txt:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Model

To fine-tune the model using Jupyter Notebook in VS Code:

1. **Open the Project in VS Code:**
   - Navigate to the project folder
   - Open VS Code from this directory or use `code .` command

2. **Open the Jupyter Notebook:**
   - Navigate to the notebook file (e.g., `qwen_finetune.ipynb`)
   - If the notebook doesn't exist, create a new one and save it with a `.ipynb` extension

3. **Run the Notebook:**
   - The notebook will contain cells for:
     - Authenticating with Hugging Face
     - Loading the Qwen 2.5 3B model and its tokenizer
     - Applying custom LoRA layers to selected modules
     - Loading and preprocessing the dataset
     - Fine-tuning the model
     - Saving the adapted LoRA parameters

4. **Execute the Cells:**
   - Run each cell sequentially using Shift+Enter or the play button
   - Monitor the output for progress and any errors

5. **Save the LoRA Weights:**
   - The final cells will save the adapted LoRA parameters to `lora_weights.pt`

## Troubleshooting

**CUDA Errors / Low Memory:**
- Ensure your GPU drivers and CUDA version are up-to-date.
- Consider reducing the batch size if memory issues occur.
- Verify that your PyTorch installation supports your CUDA version.

**Jupyter Notebook Issues:**
- If the kernel keeps dying, check your GPU memory usage and reduce batch sizes.
- Restart the kernel if you encounter persistent issues.
- Ensure your VS Code and Jupyter extensions are up to date.

**Data Loading Issues:**
- Ensure that the `q3_dataset` folder exists in the project root and contains the required datasets.
- Verify your internet connection when loading data from external sources (e.g., Hugging Face datasets).

**Dependency Problems:**
- Check that all packages in requirements.txt are correctly installed.
- Use a virtual environment (e.g., Conda) to avoid conflicts.

## Challenges Faced

During development, we encountered several challenges:

**Local Environment Setup and CUDA Errors:**
Establishing a stable local environment was challenging due to inconsistent CUDA configurations. We frequently encountered driver mismatches and runtime CUDA errors, requiring updates and sometimes using cloud-based solutions.

**Memory Constraints:**
The large Qwen 2.5 3B model imposed significant GPU memory demands. This necessitated reducing batch sizes and optimizing data loading. LoRA helped by significantly reducing the number of trainable parameters.

**Data Acquisition from Databases:**
Retrieving data from external databases was slow and sometimes inconsistent due to network issues and schema variations. We implemented robust error handling and fallback mechanisms to ensure a steady data supply.

**Integration of Quantization and Fine-Tuning Pipelines:**
Transitioning from fine-tuning to a 4-bit quantized model (.gguf format) was complex. Balancing reduced precision with maintaining model accuracy required iterative experiments.

**Dependency Management and Version Compatibility:**
Managing dependencies across PyTorch, Transformers, and CUDA was non-trivial. Ensuring version compatibility sometimes required rolling back or waiting for updated releases.

**Debugging and Reproducibility:**
Fine-tuning such a large model introduced significant debugging challenges. Extensive logging, checkpointing, and systematic troubleshooting were essential to reproduce results reliably.

## Future Enhancements

**Quantization:**
Convert the fine-tuned model to a 4-bit quantized version in .gguf format for efficient inference.

**Inference Script:**
Develop a dedicated script to serve the quantized model and integrate Retrieval-Augmented Generation (RAG) components if needed.

**Extended Evaluation:**
Implement a robust evaluation framework to assess model performance on hidden test sets derived from AI research documents.
