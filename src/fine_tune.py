from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_from_disk

# Load tokenized dataset
dataset_path = (
    "C:/Personal Projects/Prodigy Infotech/PRODIGY_GA_01/data/tokenized_wikitext"
)
dataset = load_from_disk(dataset_path)

# Load tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token


# Add labels to dataset and tokenize with padding and truncation
def add_labels(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",  # Change to max_length
        truncation=True,
        max_length=512,  # Ensure this is consistent
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


# Map the dataset to add labels and apply padding/truncation
dataset = dataset.map(add_labels, batched=True)

# Define the data collator
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding="max_length", max_length=512
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
