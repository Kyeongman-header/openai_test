import pickle
import torch
from tqdm import tqdm, trange
from dataset_consts_bart import *
import random
from torch.utils.tensorboard import SummaryWriter
import evaluate
_bleu=evaluate.load("bleu")
_real_bleu=evaluate.load("bleu")
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error Creating directory. ' + directory)
createFolder('second_level')

writer = SummaryWriter("./runs/"+"bart_wp_rake")

def do_eval(steps,whole_predictions,whole_labels):
    
    
    
    index=0

    N=1000
    
    
    print(whole_predictions[33])
    print()
    print(whole_labels[33])
    
    whole_num=0
    #whole_predictions=[]
    whole_predictions_len=len(whole_predictions)
    #whole_labels=[]
    whole_labels_len=len(whole_labels)

    whole_num=len(whole_predictions)


    print("len of sample generation : " + str(whole_num))
    print("len of label generation : " + str(len(whole_labels)))
    
    
    self_num=0

    for j in range(N if N<whole_num else whole_num):
        except_whole_labels=whole_labels[0:j]+whole_labels[j+1:N]
        _bleu.add_batch(predictions=[whole_labels[j]],references=[except_whole_labels])
        #print(except_whole_labels) 
            
        #print(_real_self_bleu)
        self_num+=1
    real_self_bleu=_bleu.compute(max_order=5)
    
    r_self_bleu_one=real_self_bleu['precisions'][0]
    r_self_bleu_bi=real_self_bleu['precisions'][1]
    r_self_bleu_tri=real_self_bleu['precisions'][2]
    r_self_bleu_four=real_self_bleu['precisions'][3]
    r_self_bleu_fif=real_self_bleu['precisions'][4]

    p_self_num=0
    for j in range(N if N<whole_num else whole_num): # 1000개에 대해서만 self-bleu.
        except_whole_predictions=whole_predictions[0:j]+whole_predictions[j+1:N]
        #self_bleu=BLEU(except_whole_predictions,weights).get_score([whole_predictions[j]])
        _real_bleu.add_batch(predictions=[whole_predictions[j]],references=[except_whole_predictions])
        
        p_self_num+=1
    
    self_bleu=_real_bleu.compute(max_order=5)
    self_bleu_one=self_bleu['precisions'][0]
    self_bleu_bi=self_bleu['precisions'][1]
    self_bleu_tri=self_bleu['precisions'][2]
    self_bleu_four=self_bleu['precisions'][3]
    self_bleu_fif=self_bleu['precisions'][4]
    
    whole_predictions_len=whole_predictions_len
    whole_labels_len=whole_labels_len
    # self_bleu_one=self_bleu_one/p_self_num
    # self_bleu_bi=self_bleu_bi/p_self_num
    # self_bleu_tri=self_bleu_tri/p_self_num
    # self_bleu_four=self_bleu_four/p_self_num
    # self_bleu_fif=self_bleu_fif/p_self_num
    # r_self_bleu_one=r_self_bleu_one/self_num
    # r_self_bleu_bi=r_self_bleu_bi/self_num
    # r_self_bleu_tri=r_self_bleu_tri/self_num
    # r_self_bleu_four=r_self_bleu_four/self_num
    # r_self_bleu_fif=r_self_bleu_fif/self_num

    print("avg prediction len : " + str(whole_predictions_len))
    print("self_bleu one : " + str(self_bleu_one))
    print("self_bleu bi : " + str(self_bleu_bi))
    print("self_bleu tri : " + str(self_bleu_tri))
    print("self_bleu four : " + str(self_bleu_four))
    print("self_bleu fif : " + str(self_bleu_fif))
    print("avg reference len : " + str(whole_labels_len))
    print("real self_bleu one : " + str(r_self_bleu_one))
    print("real self_bleu bi : " + str(r_self_bleu_bi))
    print("real self_bleu tri : " + str(r_self_bleu_tri))
    print("real self_bleu four : " + str(r_self_bleu_four))
    print("real self_bleu fif : " + str(r_self_bleu_fif))
    

    
    writer.add_scalar("self bleu bi/eval", self_bleu_bi, steps)
    writer.add_scalar("self bleu tri/eval", self_bleu_tri, steps)
    writer.add_scalar("self bleu four/eval", self_bleu_four, steps)
    writer.add_scalar("self bleu fif/eval", self_bleu_fif, steps)
    writer.add_scalar("real_self bleu bi/eval", r_self_bleu_bi, steps)
    writer.add_scalar("real_self bleu tri/eval", r_self_bleu_tri, steps)
    writer.add_scalar("real_self bleu four/eval", r_self_bleu_four, steps)
    writer.add_scalar("real_self bleu fif/eval", r_self_bleu_fif, steps)
    #writer.add_scalar("meteor",met_result,steps)
    writer.add_scalar("predictions avg len",whole_predictions_len,steps)
    writer.add_scalar("references avg len",whole_labels_len,steps)
    #writer.add_scalar("ppl",ppl.item(),steps)

import csv
import ctypes as ct
import math
import numpy as np
import pandas as pd
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

_f = pd.read_csv('bartGenerations/bart_wp_rake/test/generations_outputs_5.csv',chunksize=1000)
_f= pd.concat(_f)

num_whole_steps=len(_f.index)

first=True

count=0
par_count=0
last_keywords=""
cumul_fake_outputs=""
cumul_real_outputs=""

f=[]
r=[]
step=0
is_real_nan=False
is_real_nan_cumul=False
#progress_bar = tqdm(range(num_whole_steps))

for step, line in _f.iterrows():
    
    #if first:
    #    first=False
    #    continue
    count+=1
    #progress_bar.update(1)
    #print(line[2])
    if line[0]=='steps':
        continue

    if line[3]!=line[3]:
        is_real_nan=True
    keywords=line[2].replace('[','').replace(']','')
    fake=line[4]
    #fake=line[4].replace('[','').replace(']','').replace('<','').replace('>','').replace('newline','').replace('Newline','').replace('ewline','').replace('new line','').replace('ew line','').replace('\\x90','').replace("\\x80",'').replace('\\x9c','').replace('\\x84','').replace("\\x9d",'').replace('\\x99','').replace('\\x9','').replace('\\x8','')
    if is_real_nan is False:
        real=line[3]
        #real=line[3].replace('[','').replace(']','').replace('<','').replace('>','').replace('newline','').replace('Newline','').replace('ewline','').replace('new line','').replace('ew line','').replace('\\x90','').replace('\\x80','').replace('\\x9c','').replace('\\x84','').replace("\\x9d",'').replace('\\x99','').replace('\\x9','').replace('\\x8','')
    else:
        real=""

    if keywords==last_keywords:
        cumul_fake_outputs+=fake+". "
        cumul_real_outputs+=real+". "
        if is_real_nan:
            is_real_nan_cumul=True
        is_real_nan=False
        par_count+=1

        continue
    else:
        if count!=1 and par_count<10 and is_real_nan_cumul is False:
            f.append(cumul_fake_outputs)
            r.append(cumul_real_outputs)
        
        is_real_nan_cumul=False
        par_count=0
        cumul_fake_outputs=fake
        cumul_real_outputs=real
        last_keywords=keywords

print(count)

f.append(cumul_fake_outputs)
r.append(cumul_real_outputs)

do_eval(0,f,r)
