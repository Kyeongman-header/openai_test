import os
import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from torch.utils.data import Dataset
#from datasets import load_metric
import evaluate
#tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
pad_token_id=tokenizer.pad_token_id
import csv
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
import nltk
import random

# nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

#rouge = load_metric("rouge")
rouge = evaluate.load('rouge')
meteor= evaluate.load('meteor')
max_length=1024
batch_size=2
class Contrastive_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids,attention_mask,global_attention_mask,label):
        self.input_ids=input_ids
        self.attention_mask=attention_mask
        self.global_attention_mask=global_attention_mask
        self.label=label

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, index): 
        return (self.input_ids[index], self.attention_mask[index],self.global_attention_mask[index],self.label[index])

class MyBaseDataset(Dataset):
    def __init__(self, input_ids, attention_mask,labels,decoder_attention_mask):
        self.input_ids=input_ids
        self.attention_mask = attention_mask
        self.labels=labels
        self.decoder_attention_mask=decoder_attention_mask

    def __getitem__(self, index): 
        return {"input_ids" : self.input_ids[index], "attention_mask" : self.attention_mask[index],"decoder_input_ids" : self.labels[index][:-1],"decoder_attention_mask" :self.decoder_attention_mask[index][:-1], "labels" : self.labels[index][1:]}
        
    def __len__(self): 
        return self.input_ids.shape[0]
    

def return_dataset_2(target,source,prompt): # target을 5분할 한다.
    one_datasets=[]
    whole_datasets=[]
    for i in range(100):
        whole_datasets.append([])

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
                now_sentences+=sentences+' '
                prev+=len(t_s)
            else:
                prev=0
                split_s.append(now_sentences)
                now_sentences=""
        

        if now_sentences:
            if len(tokenizer(now_sentences).input_ids)<50 and len(split_s)!=0:
                split_s[-1]+=now_sentences
            else:
                split_s.append(now_sentences)
        
        #print(len(split_s))
        # 이렇게 하면 split_s에는 n개로 분할된 target이 있다.

        # for s in split_s:
        #     print(len(tokenizer(s).input_ids))
        if len(split_s)>=100:
            continue 
        
        input=tokenizer(source[t],max_length=200,padding="max_length",
            truncation=True,return_tensors="pt")
    
        labels=tokenizer(split_s,max_length=250,padding="max_length",
            truncation=True,return_tensors="pt")
        
        prompt_id=tokenizer(prompt[t],max_length=150,padding="max_length",
            truncation=True,return_tensors="pt").input_ids
    
        input_ids=input.input_ids.to(torch.int32)
        input_attention=input.attention_mask.to(torch.int32)
        decoder_input_ids=labels.input_ids.to(torch.int32)
        decoder_attention_mask=labels.attention_mask.to(torch.int32)
        #print(input_ids.shape)
        #print(input_attention.shape)
        #print(decoder_input_ids.shape)
        #print(prompt_id.shape)
        # input()

        """decoder_input_ids=[
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in encoder_input_ids
    ]"""
        
        whole_datasets[len(split_s)].append({"input_ids":input_ids,"input_attention":input_attention,"decoder_input_ids" : decoder_input_ids,"decoder_attention_mask":decoder_attention_mask, "prompt":prompt_id })
        one_datasets.append({"input_ids":input_ids,"input_attention":input_attention,"decoder_input_ids" : decoder_input_ids,"decoder_attention_mask":decoder_attention_mask, "prompt":prompt_id })
        
        # (30, N, data) -> n이 index, i이 데이터셋이 된다.
        # 사용시에는, (각각 순서대로의 param num에 대해서 따로 학습을 해주며, (0~30)
        # (N, data) 만 남으니까 얘네를 batch size별로 다시 묶으면
        # (N/b , b, data) 가 되고,
        # (b, data)도 사실 내부적으로는 (b, {inputids~, labels[예를 들어 5], ... }) 이런 꼴이기 때문에
        # 매번 for문을 돌면서 input_ids = (b, seq_len)
        # labels = (b, label_seq_len)
        # decoder_input_ids = (b, label_seq_len)
        # 음... 그리고 conti_prev_predictions이나 keyword_prev_predictions, memory, 같은 것도 전부 (1,~)이 아니라 (b,~)인지 shape을 확인해야.
        # 
     
    return whole_datasets,one_datasets

def return_dataset(target,source):
    labels=tokenizer(target,max_length=max_length,padding="max_length",
            truncation=True,return_tensors="pt")
    inputs=tokenizer(source,max_length=max_length,padding="max_length",
            truncation=True,return_tensors="pt")
    input_ids=inputs.input_ids
    input_attention=inputs.attention_mask
    decoder_input_ids=labels.input_ids
    decoder_attention_mask=labels.attention_mask
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
    print(decoder_input_ids.shape)
    
    return MyBaseDataset(input_ids=input_ids,attention_mask=input_attention,labels=decoder_input_ids,decoder_attention_mask=decoder_attention_mask)

# from fast_bleu import BLEU, SelfBLEU

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    bleu_score_bi=0
    bleu_score_tri=0
    bleu_score_four=0
    bleu_score_fif=0
    self_bleu_bi=0
    self_bleu_tri=0
    self_bleu_four=0
    self_bleu_fif=0
    r_self_bleu_bi=0
    r_self_bleu_tri=0
    r_self_bleu_four=0
    r_self_bleu_fif=0
    met_result=0

    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.), 'fourgram' : (1/4.,1/4.,1/4.,1/4.), 'fifthgram' : (1/5.,1/5.,1/5.,1/5.,1/5.)}
    for i in range(len(label_str)):

        ref=[label_str[i]]
        hyp=[pred_str[i]]
        bleu = BLEU(ref,weights)
        bleu_score=bleu.get_score(hyp)
        
        bleu_score_bi+=bleu_score['bigram'][0]
        bleu_score_tri+=bleu_score['trigram'][0]
        bleu_score_four+=bleu_score['fourgram'][0]
        bleu_score_fif+=bleu_score['fifthgram'][0]
        met_result += meteor.compute(predictions=hyp, references=ref)["meteor"]
        """
        self_bleu = SelfBLEU(hyp, weights).get_score()
        self_bleu_bi+=self_bleu['bigram'][0]
        self_bleu_tri+=self_bleu['trigram'][0]
        self_bleu_four+=self_bleu['fourgram'][0]
        self_bleu_fif+=self_bleu['fifthgram'][0]
        """

    bleu_score_bi=bleu_score_bi/len(label_str)
    bleu_score_tri=bleu_score_tri/len(label_str)
    bleu_score_four=bleu_score_four/len(label_str)
    bleu_score_fif=bleu_score_fif/len(label_str)
    sample_pred=random.sample(pred_str,1000)
    self_bleu = SelfBLEU(sample_pred, weights).get_score()
    real_self_bleu = SelfBLEU(sample_pred,weights).get_score()
    
    for s in range(len(self_bleu['bigram'])):

        self_bleu_bi+=self_bleu['bigram'][s]
        self_bleu_tri+=self_bleu['trigram'][s]
        self_bleu_four+=self_bleu['fourgram'][s]
        self_bleu_fif+=self_bleu['fifthgram'][s]
        r_self_bleu_bi+=real_self_bleu['bigram'][s]
        r_self_bleu_tri+=real_self_bleu['trigram'][s]
        r_self_bleu_four+=real_self_bleu['fourgram'][s]
        r_self_bleu_fif+=real_self_bleu['fifthgram'][s]

    met_result=met_result/len(label_str)
    
    self_bleu_bi=self_bleu_bi/(len(self_bleu['bigram']))
    self_bleu_tri=self_bleu_tri/(len(self_bleu['bigram']))
    self_bleu_four=self_bleu_four/(len(self_bleu['bigram']))
    self_bleu_fif=self_bleu_fif/(len(self_bleu['bigram']))
    
    r_self_bleu_bi=self_bleu_bi/(len(self_bleu['bigram']))
    r_self_bleu_tri=self_bleu_tri/(len(self_bleu['bigram']))
    r_self_bleu_four=self_bleu_four/(len(self_bleu['bigram']))
    r_self_bleu_fif=self_bleu_fif/(len(self_bleu['bigram']))

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str)
    
    return {
        "rouge1": rouge_output["rouge1"],
        "rouge2": rouge_output["rouge2"],
        "rougeL": rouge_output["rougeL"],
        "rougeLsum":rouge_output["rougeLsum"],
        "bleu_bigram":round(bleu_score_bi,4),
        "bleu_trigram":round(bleu_score_tri,4),
        "bleu_fourgram":round(bleu_score_four,4),
        "bleu_fifthgram":round(bleu_score_fif,4),
        "self_bleu_bigram":round(self_bleu_bi,4),
        "self_bleu_trigram":round(self_bleu_tri,4),
        "self_bleu_fourgram":round(self_bleu_four,4),
        "self_bleu_fifthgram":round(self_bleu_fif,4),
        "real_self_bleu_bigram":round(r_self_bleu_bi,4),
        "real_self_bleu_trigram":round(r_self_bleu_tri,4),
        "real_self_bleu_fourgram":round(r_self_bleu_four,4),
        "real_self_bleu_fifthgram":round(r_self_bleu_fif,4),
        "meteor":round(met_result,4),
    }







# dataset 전처리.
def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error Creating directory. ' + directory)

import pickle

def save_tokenize_pickle_data_1(file,total_source,total_target,last_target):
    
    createFolder("pickle_data/"+'bart'+file)
    dataset=return_dataset(total_target,total_source)
    
    print("dataset 1 making end")
    with open("pickle_data/"+'bart'+file+"/level_1.pickle","wb") as f:
        pickle.dump(dataset, f)
    
    """    
    with open("level_1_"+file+".pickle","rb") as fi:
        test = pickle.load(fi)
    """
def save_tokenize_pickle_data_2(file,total_source,total_target,last_target):
    
    createFolder("pickle_data/"+'bart_'+file)
    dataset2,one_dataset2=return_dataset_2(last_target,total_target,total_source)
    print("dataset 2 making end")
    for step,dataset in enumerate(dataset2):
        with open("pickle_data/"+'bart_'+file+"/level_2_"+str(step)+".pickle","wb") as f:
            pickle.dump(dataset,f)
    with open("pickle_data/"+'bart_'+file+"/level_2_whole.pickle","wb") as f:
            pickle.dump(one_dataset2,f)
    
