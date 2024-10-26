import torch

model.eval()

prompt = "The meaning of life is"

input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text: ", generated_text)
