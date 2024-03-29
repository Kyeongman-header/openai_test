import torch
from tqdm import tqdm, trange
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import csv
import ctypes as ct
import math
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
# load pubmed
#pubmed_test = load_dataset("scientific_papers", "pubmed", ignore_verifications=True, split="test")

# T="train"

# total_source=[]

# with open("/home/ubuntu/research/writingPrompts/"+ T +".wp_source", encoding='UTF8') as f:
#     stories = f.readlines()
#     stories = [" ".join(i.split()[0:1000]) for i in stories]
#     temp_stories=[]
#     for story in stories:
#         temp_stories.append(story.replace("<newline>",""))
#     total_source.append(temp_stories)

# total_target=[]

# with open("/home/ubuntu/research/writingPrompts/"+ T +".wp_target", encoding='UTF8') as f:
#     stories = f.readlines()
#     stories = [" ".join(i.split()[0:1000]) for i in stories]
#     temp_stories=[]
#     for story in stories:
#         temp_stories.append(story.replace("<newline>",""))
#     total_target.append(temp_stories)


# RANGE=0
# START=0 # 마지막으로 끝난 라인부터 다시 만든다.
max_length=2000

# if RANGE !=0:
#     whole_data=total_target[0][START:START+RANGE]
# else:
#     whole_data=total_target[0][START:]

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("pszemraj/led-base-book-summary")

model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/led-base-book-summary")


def generate_answer(batch):
  inputs_dict = tokenizer(batch, padding="max_length", max_length=max_length, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids
  attention_mask = inputs_dict.attention_mask
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1
  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask,max_length=512,encoder_no_repeat_ngram_size=3,no_repeat_ngram_size=3,repetition_penalty=3.5,num_beams=4,early_stopping=True,)
  return tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  """predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  print("article : ")
  print(batch["article"])
  print("predicted_abstract : ")
  print(batch["predicted_abstract"])
  return batch
  """
# f.close()
import pandas as pd
file="wp_led_results"
f = pd.read_csv(file+'.csv',chunksize=1000)
f= pd.concat(f)
count=0
first=True

for index, line in tqdm(f.iterrows()):
    #print("whole text : " + t)
    #print()
    
    if first:
        first=False
        continue
    count+=1
    print('------------------------')
    result=generate_answer(line[1])
    print("summary token len " + str(len(tokenizer(result[0]).input_ids)))
    print(line[1])
    print(result[0])
    print()
    result=generate_answer(line[3]+" " + line[1])
    print("summary token len " + str(len(tokenizer(result[0]).input_ids)))
    print(line[3] + " " + line[1])
    print(result[0])
    print()



"""result = pubmed_test.map(generate_answer, batched=True, batch_size=1)

# load rouge
rouge = load_metric("rouge")

print("Result:", rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"], rouge_types=["rouge2"])["rouge2"].mid)"""
