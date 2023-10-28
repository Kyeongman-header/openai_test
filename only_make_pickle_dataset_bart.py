import csv
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

import sys

trainvalidtest=sys.argv[1] # train
wprdbk=sys.argv[2] # wp / booksum / reedsy
bartgpt=sys.argv[3] # gpt / bart

if "gpt" in bartgpt:
    from dataset_consts_gpt import *
else:
    from dataset_consts_bart import *

total_target=[]
total_source=[]
last_target=[]

file=trainvalidtest # valid이면 밑에 file 이름 dev로 바꾸기.
f = open(file+'_'+wprdbk+'_rake_results.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
first=True

count=0


for line in rdr:
        #print(line)
        #total_target.append(print(line[1]))
        #total_source.append(line[1])
    
    if first:
        first=False
        continue
    count+=1
#    if count>100:
#        break
    last_target.append(line[0])
    total_target.append(line[1])
    total_source.append(line[2])
    print(count,end='\r')

save_tokenize_pickle_data_2(file,total_source,total_target,last_target)
