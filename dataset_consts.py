import os
import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_metric

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

import csv
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
rouge = load_metric("rouge")

max_length=1024
batch_size=2


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
    encoder_input_ids=[
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in encoder_input_ids
    ]
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




total_target=[]
total_source=[]
last_target=[]

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
    last_target.append(line[1])
    total_target.append(line[2])
    total_source.append(line[3])
    # input()



# dataset 전처리.

