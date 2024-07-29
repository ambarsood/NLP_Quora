# Quora Question Answer Dataset - NLP Project

This repository contains the code, data, and models for a comprehensive question-answering system built on the Quora Question Answer Dataset. The project involves data preprocessing, fine-tuning a GPT-2 model, and evaluating similarity-based query answering techniques.

Find the dataset : https://huggingface.co/datasets/toughdata/quora-question-answer-dataset

Find my trained model for download : https://drive.google.com/file/d/1n5JASV3MVHFPqIu67FSusumDERHrghoQ/view?usp=sharing

## Project Overview

The primary objective is to analyze and preprocess the dataset for advanced NLP applications and implement effective question-answering systems. The project is structured into three main sections:

1. **Data Preprocessing and Analysis**
2. **Fine-Tuning GPT-2**
3. **Similarity-Based Query Answering**

## Repository Structure

- **Data_Analysis/**  
  Contains scripts for data exploration, cleaning, and visualization. Includes histograms, word clouds, and n-gram analysis.

- **GPT_QuestionAnswer.ipynb**  
  Jupyter Notebook for training the GPT-2 model on the Quora dataset. Includes steps for data preparation, model fine-tuning, and generating responses.

- **gpt2-finetuned.zip**  
  The trained GPT-2 model, ready for use in generating responses.

- **qa_dataset/**  
  The dataset used in the project, formatted for easy loading and processing.

- **report.pdf**  
  Comprehensive report detailing the development processes, model performance, and findings.

- **similarity_1.py** and **similarity_2.py**  
  Scripts for implementing and evaluating similarity-based query answering models. Includes Bag of Words (BOW), Word2Vec, GloVe, and BERT approaches.

## Data Preprocessing and Analysis

### Loading and Initial Inspection

- Dataset loaded from JSONL file using pandas.
- Initial inspection reveals 56,402 entries with no missing values.

### Data Cleaning

- Duplicate entries removed.
- Missing value handling demonstrated.

### Text Preprocessing

- Tokenization, stopword removal, stemming, and lemmatization applied.

### Data Analysis

- Visualizations include question length distribution, word clouds, and n-gram analysis.

## Fine-Tuning GPT-2

### Dataset Preparation

- Data loaded from CSV, combined into a text string, and formatted for Hugging Face Dataset.

### Model Fine-Tuning

- GPT-2 tokenizer and model loaded, training configured, and model trained and saved.

### Model Interaction

- Interface created using Gradio for user interaction.

### Model Evaluation

- Evaluated using ROUGE, BLEU, and F1-score metrics.

## Similarity-Based Query Answering

### Introduction

- **Objective:** Match queries with the closest predefined question based on semantic similarity.

### Models and Techniques

- **Bag of Words (BOW)**
- **Word2Vec**
- **GloVe**
- **BERT**

### Comparison of Approaches

- Analysis of the performance and computational efficiency of each approach.

### Conclusion

- **BERT** provides the best performance for complex queries, while **BOW**, **Word2Vec**, and **GloVe** offer varying balances of performance and efficiency.
