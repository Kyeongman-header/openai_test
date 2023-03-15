import csv
import pandas as pd
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from transformers import AutoTokenizer
import random
tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large")

file="wp_led_results"
f = pd.read_csv(file+'.csv',chunksize=1000)
f= pd.concat(f)

dataset=[]

first=True
lt_avg=0
lt_max=0
lt_min=999999
lt_len_arr=[]
num=0

target_avg=0
target_max=0
target_min=999999
target_len_arr=[]

source_avg=0
source_max=0
source_min=999999
source_len_arr=[]

for index, line in tqdm(f.iterrows()):
    
    if first:
        first=False
        continue
    lt_len=len(tokenizer(line[1]).input_ids)
    target_len=len(tokenizer(line[2]).input_ids)
    source_len=len(tokenizer(line[3]).input_ids)
    lt_max=max(lt_len,lt_max)
    lt_min=min(lt_min,lt_len)
    lt_avg+=int(lt_len)
    target_max=max(target_len,target_max)
    target_min=min(target_min,target_len)
    target_avg+=int(target_len)
    source_max=max(source_len,source_max)
    source_min=min(source_min,source_len)
    source_avg+=int(source_len)

    num+=1
    dataset.append({'target':line[1],'arb_sum':line[2],'source':line[3]})

def report():
    print("num " + str(num))
    print("target avg " + str(lt_avg/num))
    print("target max " + str(lt_max))
    print("target min " + str(lt_min))


    print("arb_sum avg " + str(target_avg/num))
    print("arb_sum max " + str(target_max))
    print("arb_sum min " + str(target_min))

    print("source avg " + str(source_avg/num))
    print("source max " + str(source_max))
    print("source min " + str(source_min))

report()

for data in tqdm(random.sample(dataset,16)):
    print('-----------------------')
    print(data['source'])
    print()
    print(data['arb_sum'])
    print()
    print(data['target'])
    