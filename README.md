# Probing Internal Activations and Representation Engineering for Mathematical Reasoning

This repository contains the term project for **Fall 2024 - CS 546: Advanced Topics in NLP (Section ATN)**. The project explores probing and representation engineering techniques to analyze and improve mathematical reasoning in large language models (LLMs).

## Overview

The focus of this project is on:
- Analyzing internal activations of LLMs during mathematical reasoning tasks.
- Evaluating the feasibility of representation engineering to improve reasoning accuracy.

The experiments were conducted primarily on lightweight LLMs like Llama 3.2-3B and Mistral 7B, with the GSM8K dataset as the core benchmark.

## Key Features
- Implementation of probing techniques (logistic regression, SVM) for identifying incorrect reasoning.
- Contrastive learning datasets generated using Llama3 and GPT-4.
- Visualization of internal representations through PCA and other metrics.
- Evaluation of reasoning accuracy via logits and hidden state analysis.

## Tutorial

For a detailed step-by-step demonstration, we recommend running the following notebook:

**`probe-hidden-states-llama-3.2-3B-Instruct-gsm8k-performance.ipynb`**

This notebook provides a comprehensive overview of the probing process and model evaluation, making it a great starting point for understanding the methodology and results.