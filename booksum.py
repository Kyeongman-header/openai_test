import csv
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
import matplotlib.pyplot as plt
from dataset_consts import *





count=0


file="booksum"

encoder_max_length = 1024
decoder_max_length = 1024
batch_size = 1

f = open(file+'.csv', 'r', encoding='utf-8') # 이 아래의 부분을 함수화하려고 했는데, 어째서인지 함수로 만들면 9,600 line까지만 처리된다.

rdr = csv.reader(f)


first=True

text=[]
summaries=[]
analysis=[]

text_avg=0
text_max=0
text_min=999999
text_len_arr=[]
num=0

sum_avg=0
sum_max=0
sum_min=999999
sum_len_arr=[]

ana_avg=0
ana_max=0
ana_min=999999
ana_len_arr=[]

sum_ana_avg=0
sum_ana_max=0
sum_ana_min=999999
sum_ana_len_arr=[]

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

    chapter=line[9]
    chapter_len=int(float(line[10]))
    summary=line[13]
    anal=line[14]
    summary_len=int(float(line[15]))
    anal_len=int(float(line[16]))
    
    sum_len_arr.append(summary_len)
    ana_len_arr.append(anal_len)
    text_len_arr.append(chapter_len)
    sum_ana_len_arr.append(anal_len+summary_len)

    num+=1
    # print("summary : ")
    # print(summary)
    # print("len : ")
    # print(summary_len)
    sum_max=max(sum_max,summary_len)
    sum_min=min(sum_min,summary_len)
    sum_avg+=int(summary_len)

    # print("analysis : ")
    # print(anal)
    # print("len : ")
    # print(anal_len)

    ana_max=max(ana_max,anal_len)
    ana_min=min(ana_min,anal_len)
    ana_avg+=int(anal_len)


    sum_ana_max=max(sum_ana_max,anal_len+summary_len)
    sum_ana_min=min(sum_ana_min,anal_len+summary_len)
    sum_ana_avg+=int(summary_len+anal_len)
    
    # print("text : ")
    # print(chapter)
    # print("len : ")
    # print(chapter_len)
    text_max=max(text_max,chapter_len)
    text_min=min(text_min,chapter_len)
    text_avg+=int(chapter_len)

    text.append(chapter)
    summaries.append(summary)
    analysis.append(anal)



def report():
    print("text avg " + str(text_avg/num))
    print("text max " + str(text_max))
    print("text min " + str(text_min))


    print("summary avg " + str(sum_avg/num))
    print("summary max " + str(sum_max))
    print("summary min " + str(sum_min))
    
    print("analysis avg " + str(ana_avg/num))
    print("analysis max " + str(ana_max))
    print("analysis min " + str(ana_min))
    
    print("sum+anal avg " + str(sum_ana_avg/num))
    print("sum+anal max " + str(sum_ana_max))
    print("sum+anal min " + str(sum_ana_min))

# if(plot):
#     plt.subplot(221)
#     plt.hist(sum_len_arr,bins=100)
#     plt.title('summary_length')
#     plt.subplot(222)
#     plt.hist(ana_len_arr,bins=100)
#     plt.title('analysis_length')
#     plt.subplot(223)
#     plt.hist(text_len_arr,bins=100)
#     plt.title('text_length')
#     plt.subplot(224)
#     plt.hist(sum_ana_len_arr,bins=100)
#     plt.title('summary_analysis_length')
#     plt.show()

    
report()

 


# text,summaries,analysis,num=loader(plot=False)

prompts=[tokenizer.pad_token] * num

whole_datasets=return_dataset_2(text,summaries,prompts)

createFolder("pickle_data/"+file)
with open("pickle_data/"+file+"/level_2.pickle","wb") as f:
    pickle.dump(whole_datasets,f)