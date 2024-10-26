from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained("./deployed_fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./deployed_fine_tuned_gpt2")

tokenizer.pad_token = tokenizer.eos_token


@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    generated_outputs = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
