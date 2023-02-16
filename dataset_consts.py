import os
import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_metric

tokenizer = AutoTokenizer.from_pretrained("t5-base")

import csv
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

rouge = load_metric("rouge")

max_length=1024
batch_size=4


class MyBaseDataset(Dataset):
    def __init__(self, input_ids, attention_mask,labels):
        self.input_ids=input_ids
        self.attention_mask = attention_mask
        self.labels=labels

    def __getitem__(self, index): 
        return {"input_ids" : self.input_ids[index], "attention_mask" : self.attention_mask[index],"decoder_input_ids" : self.labels[index],"labels" : self.labels[index]}
        
    def __len__(self): 
        return self.input_ids.shape[0]
    

def return_dataset_2(target,source,prompt): # target을 5분할 한다.
    whole_dataset=[]
    for t in trange(len(target)):
        whole_len=len(tokenizer(target[t]).input_ids)
        #print(whole_len)
        sentences_in_target=sent_tokenize(target[t])
        
        prev=0
        
        split_s=[]
        now_sentences=""
        
        for sentences in sentences_in_target:
            t_s=tokenizer(sentences).input_ids
            if len(t_s)+prev<=200:
                now_sentences+=sentences
                prev+=len(t_s)
            else:
                prev=0
                split_s.append(now_sentences)
                now_sentences=""

        if now_sentences:
            if len(tokenizer(now_sentences).input_ids)<50:
                split_s[-1]+=now_sentences
            else:
                split_s.append(now_sentences)
        
        #print(len(split_s))
        # 이렇게 하면 split_s에는 n개로 분할된 target이 있다.

        # for s in split_s:
        #     print(len(tokenizer(s).input_ids))

        input=tokenizer(source[t],max_length=max_length-100,padding="max_length",
            truncation=True,return_tensors="pt")
    
        labels=tokenizer(split_s,max_length=max_length-100,padding="max_length",
            truncation=True,return_tensors="pt")
        
        prompt_id=tokenizer(prompt[t],return_tensors="pt").input_ids
    
        input_ids=input.input_ids
        input_attention=input.attention_mask
        decoder_input_ids=labels.input_ids
        #print(input_ids.shape)
        #print(input_attention.shape)
        #print(decoder_input_ids.shape)
        #print(prompt_id.shape)
        # input()

        """decoder_input_ids=[
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in encoder_input_ids
    ]"""
    
        whole_dataset.append({"input_ids":input_ids,"input_attention":input_attention,"decoder_input_ids" : decoder_input_ids,"prompt":prompt_id })
     
    return whole_dataset

def return_dataset(target,source):
    labels=tokenizer(target,max_length=max_length,padding="max_length",
            truncation=True,return_tensors="pt")
    inputs=tokenizer(source,max_length=max_length,padding="max_length",
            truncation=True,return_tensors="pt")
    input_ids=inputs.input_ids
    input_attention=inputs.attention_mask
    encoder_input_ids=inputs.input_ids
    """encoder_input_ids=torch.LongTensor([
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in encoder_input_ids
    ])
    """
    #input_ids=torch.reshape(input_ids,(-1,batch_size,max_length))
    #input_attention=torch.reshape(input_attention,(-1,batch_size,max_length))
    #encoder_input_ids=torch.reshape(encoder_input_ids,(-1,batch_size,max_length))

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







# dataset 전처리.
def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error Creating directory. ' + directory)

import pickle

def save_tokenize_pickle_data(file,total_source,total_target,last_target):
    
    createFolder("pickle_data/"+file)
    dataset=return_dataset(total_target,total_source)


    with open("pickle_data/"+file+"/level_1.pickle","wb") as f:
        pickle.dump(dataset, f)
    
    """    
    with open("level_1_"+file+".pickle","rb") as fi:
        test = pickle.load(fi)
    """
    
    dataset2=return_dataset_2(last_target,total_target,total_source)
    with open("pickle_data/"+file+"/level_2.pickle","wb") as f:
        pickle.dump(dataset2,f)
    