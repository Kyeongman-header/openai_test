import os
import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import AutoConfig
from torch.utils.data import Dataset
from datasets import load_metric

config = AutoConfig.from_pretrained('facebook/bart-large-cnn')
model =  AutoModel.from_config(config) # not pretrained.
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

import csv
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

class MyBaseDataset(Dataset):
    def __init__(self, input_ids, attention_mask,labels):
        self.input_ids=input_ids
        self.attention_mask = attention_mask
        self.labels=labels

    def __getitem__(self, index): 
        return self.input_ids[index], self.attention_mask[index],self.labels[index]
        
    def __len__(self): 
        return self.input_ids.shape[0]

def return_dataset(target,source):
    labels=tokenizer(target,max_length=max_length,padding="max_length",
            truncation=True,return_tensors="pt")
    inputs=tokenizer(source,max_length=max_length,padding="max_length",
            truncation=True,return_tensors="pt")
    input_ids=inputs.input_ids
    input_attention=inputs.attention_mask
    encoder_input_ids=inputs.input_ids
    input_ids=torch.reshape(input_ids,(-1,batch_size,max_length))
    input_attention=torch.reshape(input_attention,(-1,batch_size,max_length))
    encoder_input_ids=torch.reshape(encoder_input_ids,(-1,batch_size,max_length))

    print(input_ids.shape)
    print(input_attention.shape)
    print(encoder_input_ids.shape)
    
    return MyBaseDataset(input_ids,input_attention,encoder_input_ids)

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

batch_size=2
max_length=1024
TRAIN_RANGE=25000


total_target=[]
total_source=[]

f = open('wp_led_results.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)    
first=True


for line in rdr:
        #print(line)
        #total_target.append(print(line[1]))
        #total_source.append(line[1])
    if first:
        first=False
        continue
    # print(line[0])
    # print(line[1])
    # print(line[2])
    # print(line[3])
    total_target.append(line[2])
    total_source.append(line[3])
    # input()

total_target=total_target[:TRAIN_RANGE]
total_source=total_source[:TRAIN_RANGE]
val_total_target=total_target[TRAIN_RANGE:]
val_total_source=total_source[TRAIN_RANGE:]

train_dataset=return_dataset(total_target,total_source)
valid_dataset=return_dataset(val_total_target,val_total_source)

# dataset 전처리.



from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer

rouge = load_metric("rouge")

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    fp16_backend="apex",
    output_dir="./",
    logging_steps=250,
    eval_steps=5000,
    save_steps=500,
    warmup_steps=1500,
    save_total_limit=2,
    gradient_accumulation_steps=4,
)



model.config.num_beams = 4
model.config.max_length = 512
model.config.min_length = 100
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()

# model.save_pretrained('./first_level_best.pt')