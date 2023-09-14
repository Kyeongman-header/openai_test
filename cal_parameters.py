from transformers import BartTokenizer, BartForConditionalGeneration
"""
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
"""
from transformers import AutoModelForCausalLM

model_name="gpt2-medium"
model=AutoModelForCausalLM.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in {model_name}: {total_params}")
