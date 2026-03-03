from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_id)

print("Saving locally...")
tokenizer.save_pretrained("./models/tinyllama")
model.save_pretrained("./models/tinyllama")

print("✅ Model saved cleanly in ./models/tinyllama")