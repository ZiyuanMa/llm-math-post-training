# Unlocking Mathematical Reasoning in Small Language Models with Post-Training

This project investigates the viability of Reinforcement Learning from Human Feedback (RLHF) on a relatively small-scale model—the Qwen2.5-1.5B-Instruct, which comprises only 1.5 billion parameters—with a focus on enhancing mathematical reasoning tasks. We specifically evaluate Group-based Regularized Policy Optimization (GRPO), an efficient RLHF variant, and compare its performance against standard Supervised Fine-tuning (SFT).

## Project Overview

Both fine-tuning strategies (GRPO and SFT) were applied independently to the same baseline model. Unlike previous studies emphasizing algorithmic variety, this research prioritizes understanding the performance differences between these two distinct fine-tuning approaches when applied to an identical model and for the same set of tasks.

Model performance is evaluated using two distinct mathematical reasoning benchmarks:
1. A synthetic dataset containing structured systems of linear equations with guaranteed solvability
2. The GSM8K dataset consisting of real-world grade-school-level arithmetic word problems

## Repository Structure

- `sft.py`: Implementation of supervised fine-tuning approach
- `grpo.py`: Implementation of GRPO (RLHF variant) fine-tuning approach
- `data.py`: Generates the synthetic linear equation dataset
- `utils.py`: Helper functions for parsing and evaluating model outputs
- `linear_equation_eval.py`: Evaluation script for the linear equations benchmark
- `gsm8k_eval.py`: Evaluation script for the GSM8K benchmark
- `sft.ipynb`/`grpo.ipynb`: Jupyter notebooks for the corresponding training methods

### Dataset and Evaluation Files
- `data.json`: Synthetic linear equations dataset
- `baseline_linear_equation.json`/`baseline_gsm8k.json`: Baseline model evaluations
- `sft_linear_equation.json`/`sft_gsm8k.json`: SFT model evaluations
- `grpo_linear_equation.json`/`grpo_gsm8k.json`: GRPO model evaluations

## Model and Fine-tuning

The project uses **Qwen2.5-1.5B-Instruct** as the base model, a relatively small LLM with only 1.5 billion parameters. We compare two fine-tuning methods:

1. **Supervised Fine-tuning (SFT)**:
   - Traditional approach that directly fine-tunes on examples with ground truth responses
   - Implemented with Unsloth LoRA for efficient training

2. **Group-based Regularized Policy Optimization (GRPO)**:
   - An efficient RLHF variant that optimizes based on reward signals
   - Employs a correctness-based reward function to guide model learning
   - Particularly adapted for mathematical reasoning tasks

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Unsloth
- Hugging Face Transformers
- TRL (Transformer Reinforcement Learning)

### Running the Code

1. Generate synthetic dataset (if needed):
   ```
   python data.py
   ```

2. Run SFT training:
   ```
   python sft.py
   ```

3. Run GRPO training:
   ```
   python grpo.py
   ```

4. Evaluate on linear equations:
   ```
   python linear_equation_eval.py
   ```

5. Evaluate on GSM8K:
   ```
   python gsm8k_eval.py
   ```

## Goals

The project aims to determine whether RLHF (specifically GRPO) can substantially improve mathematical reasoning capabilities in small language models compared to traditional SFT approaches. Results are analyzed across both the synthetic linear equations task and the more complex GSM8K benchmark.

Key findings:

- GRPO significantly enhances mathematical reasoning in small models, with superior generalization compared to SFT
- SFT excels at specific task adaptation but shows limited generalization with specialized data
- RLHF achieves better balance between task-specific performance and broader capabilities

### Accuracy Results

#### Linear Equations (In-domain task)
- Baseline: 4.2%
- SFT: 71.6%
- GRPO: 74.0%

#### GSM8K (Out-of-domain task)
- Baseline: 55.19%
- SFT: 0.73%
- GRPO: 64.44% 