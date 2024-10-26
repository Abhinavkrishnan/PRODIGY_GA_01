from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_gpt2"  # Path where the fine-tuned model is saved
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
