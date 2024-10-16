# src/generate_text.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./models')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text
input_text = "Your prompt here"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(input_ids, max_length=100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
