import csv
import pandas as pd
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
import matplotlib.pyplot as plt
from dataset_consts import *




count=0



encoder_max_length = 1024
decoder_max_length = 1024
batch_size = 1

file="ROCStories_train"


f = open(file+'.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)


first=True

stories=[]
titles=[]

text_avg=0
text_max=0
text_min=999999
text_len_arr=[]
num=0

prompt_avg=0
prompt_max=0
prompt_min=999999
prompt_len_arr=[]

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

    #text=line[2]+" "+line[3]+" "+line[4]+" "+line[5]+" "+line[6]
    text=line[3]+" "+line[4]+" "+line[5]+" "+line[6]
    text_len=len(tokenizer(text).input_ids)

    prompt=line[1] + ". " + line[2]
    prompt_len=len(tokenizer(prompt).input_ids)
    
    text_len_arr.append(text_len)
    prompt_len_arr.append(prompt_len)

    num+=1
    # print("summary : ")
    # print(summary)
    # print("len : ")
    # print(summary_len)
    prompt_max=max(prompt_max,prompt_len)
    prompt_min=min(prompt_min,prompt_len)
    prompt_avg+=int(prompt_len)

    # print("analysis : ")
    # print(anal)
    # print("len : ")
    # print(anal_len)

    text_max=max(text_max,text_len)
    text_min=min(text_min,text_len)
    text_avg+=int(text_len)
    
    # print("text : ")
    # print(chapter)
    # print("len : ")
    # print(chapter_len)

    stories.append(text)
    titles.append(prompt)
    


def report():
    print("num " + str(num))
    print("text avg " + str(text_avg/num))
    print("text max " + str(text_max))
    print("text min " + str(text_min))


    print("title avg " + str(prompt_avg/num))
    print("title max " + str(prompt_max))
    print("title min " + str(prompt_min))

# if(plot):
#     plt.subplot(211)
#     plt.hist(text_len_arr,bins=100)
#     plt.title('story_length')
#     plt.subplot(212)
#     plt.hist(prompt_len_arr,bins=100)
#     plt.title('title_length')
#     plt.show()

    
report()




# stories,titles,num=loader(plot=True)

whole_datasets=return_dataset(stories[:50000],titles[:50000])
#file="ROCStories_valid"

createFolder("pickle_data/"+file)
with open("pickle_data/"+file+"/level_1.pickle","wb") as f:
    pickle.dump(whole_datasets,f)
