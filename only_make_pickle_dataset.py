import csv
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
from dataset_consts import *

total_target=[]
total_source=[]
last_target=[]

file="train" # valid이면 밑에 file 이름 바꿔주기.
f = open('wp_led_results.csv', 'r', encoding='utf-8')
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
#    count+=1
#    if count>100:
#        break
    # print(line[0])
    # print(line[1])
    # print(line[2])
    # print(line[3])
    last_target.append(line[1])
    total_target.append(line[2])
    total_source.append(line[3])
    # input()

save_tokenize_pickle_data(file,total_source[:10],total_target[:10],last_target[:10])
