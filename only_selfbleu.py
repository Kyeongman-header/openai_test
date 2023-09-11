import pickle
import torch
from tqdm import tqdm, trange
from dataset_consts_bart import *
import random
from torch.utils.tensorboard import SummaryWriter
import evaluate
import nltk.translate.bleu_score as bleu
from nltk.tokenize import TweetTokenizer
_bleu=evaluate.load("bleu")
_real_bleu=evaluate.load("bleu")
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error Creating directory. ' + directory)
createFolder('second_level')

writer = SummaryWriter("./runs/"+"wp1000")

def do_eval(steps,whole_predictions,whole_labels):
    #for label in whole_labels[:3]:
    #    print(label)
    
    
    index=0

    N=1000
    
    
    #print(whole_predictions[33])
    #print()
    #print(whole_labels[33])
    
    whole_num=0
    #whole_predictions=[]
    whole_predictions_len=len(whole_predictions)
    #whole_labels=[]
    whole_labels_len=len(whole_labels)

    whole_num=len(whole_predictions)


    print("len of sample generation : " + str(whole_num))
    print("len of label generation : " + str(len(whole_labels)))
    
    
    self_num=0
    print("prediction bleu")
    self_bleu_one=0
    self_bleu_bi=0
    self_bleu_tri=0
    self_bleu_four=0
    self_bleu_fif=0

    for j in trange(N if N<len(whole_predictions) else len(whole_predictions)): # 1000개에 대해서만 self-bleu.
        except_whole_predictions=whole_predictions[0:j]+whole_predictions[j+1:N]
        #self_bleu=BLEU(except_whole_predictions,weights).get_score([whole_predictions[j]])

        #print([except_whole_predictions])
        #print(whole_predictions[j])
        refs=[]
        for predicts in except_whole_predictions:

            refs.append(TweetTokenizer(predicts))

        print(refs)
        hyp=TweetTokenizer(whole_predictions[j])
        print(hyp)
        self_bleu=bleu.sentence_bleu(refs,hyp,weights=[(1./2.,1./2.),(1./3.,1./3.,1./3.),(1./4.,1./4.,1./4.,1./4.),(1./5.,1./5.,1./5.,1./5.,1./5.)])
        print(self_bleu)
        self_bleu_bi+=self_bleu[0]
        self_bleu_tri+=self_bleu[1]
        self_bleu_four+=self_bleu[2]
        self_bleu_fif+=self_bleu[3]
        """
        self_bleu_one+=_bleu.compute(predictions=[whole_predictions[j]],references=[except_whole_predictions],max_order=1)['bleu']
        #print(self_bleu_one)
        self_bleu_bi+=_bleu.compute(predictions=[whole_predictions[j]],references=[except_whole_predictions],max_order=2)['bleu']
        #print(self_bleu_bi)
        self_bleu_tri+=_bleu.compute(predictions=[whole_predictions[j]],references=[except_whole_predictions],max_order=3)['bleu']
        #print(self_bleu_tri)
        self_bleu_four+=_bleu.compute(predictions=[whole_predictions[j]],references=[except_whole_predictions],max_order=4)['bleu']
        #print(self_bleu_four)
        self_bleu_fif+=_bleu.compute(predictions=[whole_predictions[j]],references=[except_whole_predictions],max_order=5)['bleu']
        #print(self_bleu_fif)
        """


    # real_self_bleu=_bleu.compute(max_order=5)
    print("compute complete")


    #print("self_bleu one : " + str(self_bleu_one/len(whole_predictions)))
    print("self_bleu bi : " + str(self_bleu_bi/len(whole_predictions)))
    print("self_bleu tri : " + str(self_bleu_tri/len(whole_predictions)))
    print("self_bleu four : " + str(self_bleu_four/len(whole_predictions)))
    print("self_bleu fif : " + str(self_bleu_fif/len(whole_predictions)))
    
    r_self_bleu_one=0
    r_self_bleu_bi=0
    r_self_bleu_tri=0
    r_self_bleu_four=0
    r_self_bleu_fif=0

    for j in trange(len(whole_labels)): # 1000개에 대해서만 self-bleu.
        except_whole_labels=whole_labels[0:j]+whole_labels[j+1:N]
    #self_bleu=BLEU(except_whole_predictions,weights).get_score([whole_predictions[j]])
        refs=[]
        for predicts in except_whole_labels:

            refs.append(TweetTokenizer(predicts))

        print(refs)
        hyp=TweetTokenizer(whole_labels[j])
        print(hyp)
        self_bleu=bleu.sentence_bleu(refs,hyp,weights=[(1./2.,1./2.),(1./3.,1./3.,1./3.),(1./4.,1./4.,1./4.,1./4.),(1./5.,1./5.,1./5.,1./5.,1./5.)])
        print(self_bleu)
        r_self_bleu_bi+=self_bleu[0]
        r_self_bleu_tri+=self_bleu[1]
        r_self_bleu_four+=self_bleu[2]
        r_self_bleu_fif+=self_bleu[3]
        """
        r_self_bleu_one+=_bleu.compute(predictions=[whole_labels[j]],references=[except_whole_labels],max_order=1)['bleu']
        #print(self_bleu_one)
        r_self_bleu_bi+=_bleu.compute(predictions=[whole_labels[j]],references=[except_whole_labels],max_order=2)['bleu']
        #print(self_bleu_bi)
        r_self_bleu_tri+=_bleu.compute(predictions=[whole_labels[j]],references=[except_whole_labels],max_order=3)['bleu']
        #print(self_bleu_tri)
        r_self_bleu_four+=_bleu.compute(predictions=[whole_labels[j]],references=[except_whole_labels],max_order=4)['bleu']
        #print(self_bleu_four)
        r_self_bleu_fif+=_bleu.compute(predictions=[whole_labels[j]],references=[except_whole_labels],max_order=5)['bleu']
        #print(self_bleu_fif)
        """


    # real_self_bleu=_bleu.compute(max_order=5)
    print("compute complete")


    #print("r_self_bleu one : " + str(r_self_bleu_one/len(whole_labels)))
    print("r_self_bleu bi : " + str(r_self_bleu_bi/len(whole_labels)))
    print("r_self_bleu tri : " + str(r_self_bleu_tri/len(whole_labels)))
    print("r_self_bleu four : " + str(r_self_bleu_four/len(whole_labels)))
    print("r_self_bleu fif : " + str(r_self_bleu_fif/len(whole_labels)))
    """
    for j in range(N if N<whole_num else whole_num):
        except_whole_labels=whole_labels[0:j]+whole_labels[j+1:N]
        _bleu.add_batch(predictions=[whole_labels[j]],references=[except_whole_labels])
        #print(except_whole_labels) 
            
        #print(_real_self_bleu)
        self_num+=1
    print("add batch complete")
    real_self_bleu=_bleu.compute(max_order=5)
    print("compute complete")
    
    r_self_bleu_one=real_self_bleu['precisions'][0]
    r_self_bleu_bi=real_self_bleu['precisions'][1]
    r_self_bleu_tri=real_self_bleu['precisions'][2]
    r_self_bleu_four=real_self_bleu['precisions'][3]
    r_self_bleu_fif=real_self_bleu['precisions'][4]
    print("self_bleu one : " + str(r_self_bleu_one))
    print("self_bleu bi : " + str(r_self_bleu_bi))
    print("self_bleu tri : " + str(r_self_bleu_tri))
    print("self_bleu four : " + str(r_self_bleu_four))
    print("self_bleu fif : " + str(r_self_bleu_fif))
    """
    
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

    #writer.add_scalar("self bleu one/eval", self_bleu_one/whole_predictions_len, steps)
    writer.add_scalar("self bleu bi/eval", self_bleu_bi/whole_predictions_len, steps)
    writer.add_scalar("self bleu tri/eval", self_bleu_tri/whole_predictions_len, steps)
    writer.add_scalar("self bleu four/eval", self_bleu_four/whole_predictions_len, steps)
    writer.add_scalar("self bleu fif/eval", self_bleu_fif/whole_predictions_len, steps)
    #writer.add_scalar("real_self bleu one/eval", r_self_bleu_one/whole_labels_len, steps)
    writer.add_scalar("real_self bleu bi/eval", r_self_bleu_bi/whole_labels_len, steps)
    writer.add_scalar("real_self bleu tri/eval", r_self_bleu_tri/whole_labels_len, steps)
    writer.add_scalar("real_self bleu four/eval", r_self_bleu_four/whole_labels_len, steps)
    writer.add_scalar("real_self bleu fif/eval", r_self_bleu_fif/whole_labels_len, steps)
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

_f = pd.read_csv('bartGenerations/bart_wp_1000_rake/test/generations_outputs_5.csv',chunksize=1000)

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
    #print(real)
    #input()
    if keywords==last_keywords:
        cumul_fake_outputs+=fake+". "
        cumul_real_outputs+=real+". "
        if is_real_nan:
            is_real_nan_cumul=True
        is_real_nan=False
        par_count+=1

        continue
    else:
        if count!=1 and is_real_nan_cumul is False:
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
