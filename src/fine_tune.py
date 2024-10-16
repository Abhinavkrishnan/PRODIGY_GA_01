# src/fine_tune.py
import os
from transformers import GPT2Tokenizer

# Load the dataset
dataset_path = os.path.join('data', 'your_dataset.txt')
with open(dataset_path, 'r', encoding='utf-8') as f:
    text_data = f.read()

# Tokenization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = tokenizer(text_data, return_tensors='pt', truncation=True, max_length=512)
