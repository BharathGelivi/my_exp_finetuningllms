from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=5)

print(tokenizer.decode(outputs[0]))