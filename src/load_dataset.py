from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load the wikitext dataset and only take a subset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

# Reduce dataset size (e.g., take the first 1000 samples)
ds["train"] = ds["train"].select(range(1000))
ds["validation"] = ds["validation"].select(range(100))  # For evaluation, keep it small
ds["test"] = ds["test"].select(range(100))  # For testing

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add a padding token (use eos_token as padding token)
tokenizer.pad_token = tokenizer.eos_token


# Tokenization function with padding and truncation
def tokenize_function(examples):
    if "text" in examples:
        return tokenizer(
            examples["text"],
            padding=True,  # Apply padding to ensure consistent sequence length
            truncation=True,  # Truncate sequences longer than the model's max length
            max_length=512,  # Adjust max_length if needed
        )
    else:
        return {}


# Apply the tokenization function to the dataset with a specified batch size
tokenized_dataset = ds.map(tokenize_function, batched=True, batch_size=100)

# Print the first tokenized example to verify
print(tokenized_dataset["train"][0])

# Save the tokenized dataset to disk
tokenized_dataset.save_to_disk("data/tokenized_wikitext")
print("Tokenized dataset saved to disk!")
