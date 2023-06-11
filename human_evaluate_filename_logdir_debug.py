"""
총 3개의 질문 사항에 대해 각각 5가지 보기가 있다.
1. 이 글은 주제의 통일성이 마치 사람이 쓴 것과 같다.
-> 1 매우 그렇다 2 그렇다 3 보통이다 4 아니다 5 매우 아니다
2. 이 글은 한편의 글로써 완결성이 마치 사람이 쓴 것과 같다.
-> 1 매우 그렇다 2 그렇다 3 보통이다 4 아니다 5 매우 아니다
3. 이 글은 사람이 쓴 것 같다.
-> 1 매우 그렇다 2 그렇다 3 보통이다 4 아니다 5 매우 아니다
"""

from tqdm import tqdm, trange
import random
import sys


testfile_name=sys.argv[1] # 예제 : wp_all_generations_outputs
log_dir=sys.argv[2] # coh1
debug=int(sys.argv[3]) # 1 or 0

if debug==1:
    debug=True
else:
    debug=False

print("test file : " + testfile_name + ".csv")
print("log dir : " + log_dir)
print("debug mode : " + str(debug))


import csv
import ctypes as ct
import math
import numpy as np
import pandas as pd
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

_f = pd.read_csv(testfile_name+'.csv',chunksize=1000)
_f= pd.concat(_f)

num_whole_steps=len(_f.index)

first=True

count=0
last_keywords=""
cumul_fake_outputs=""
cumul_real_outputs=""

f=[]
r=[]
step=0

#progress_bar = tqdm(range(num_whole_steps))

for step, line in _f.iterrows():
    
    if first:
        first=False
        continue
    count+=1
    #progress_bar.update(1)
    
    keywords=line[2].replace('[','').replace(']','')
    fake=line[4].replace('[','').replace(']','')
    real=line[3].replace('[','').replace(']','')
    if debug:
        print("keywords : " + line[2].replace('[','').replace(']',''))
        print("fake outputs : " + line[4].replace('[','').replace(']',''))
        print("real outputs : " + line[3].replace('[','').replace(']',''))
        input()

    if keywords==last_keywords:
        cumul_fake_outputs+=fake
        cumul_real_outputs+=real
        continue
    else:
        if count!=1:
            f.append({"text" : cumul_fake_outputs , "label" : "fake"})
            r.append({"text" : cumul_real_outputs, "label" : "real"})
            
            if debug:
                print("eval results : " )
                print("fake : " + cumul_fake_outputs)
                print("real : " + cumul_real_outputs)
                print("keywords : " + last_keywords)
                print("fake score : ")
                
                print("real score : ")
                
                print("###############")
            
        cumul_fake_outputs=fake
        cumul_real_outputs=real
        last_keywords=keywords


f.append({"text" : cumul_fake_outputs , "label" : "fake"})
r.append({"text" : cumul_real_outputs, "label" : "real"})
step+=1
random.shuffle(f)
random.shuffle(r)
mix=f+r
random.shuffle(mix)

f_scores_a=[]
f_scores_b=[]
f_scores_c=[]
r_scores_a=[]
r_scores_b=[]
r_scores_c=[]
for i,sample in enumerate(mix):
    print("\n\n" + str(i) + ". The sample text from " + testfile_name +". \n ############################")
    print(sample["text"])
    
    while True:
        print("Question 1. 이 글은 주제의 통일성이 마치 사람이 쓴 것과 같다.")
        print("1. 매우 그렇다. 2. 그렇다. 3. 보통이다. 4. 아니다. 5. 매우 아니다.")
        print("Answer the number.",end=" ")
        a=input()
        if a.isdigit() and int(a)<=5 and int(a)>0:
            break
        else:
            print("You answerd wrong case => " + a)
            print("Please answer the question again.")

    while True:
        print("Question 2. 이 글은 한편의 글로써 완결성이 마치 사람이 쓴 것과 같다.")
        print("1. 매우 그렇다. 2. 그렇다. 3. 보통이다. 4. 아니다. 5. 매우 아니다.")
        print("Answer the number.",end=" ")
        b=input()
        if b.isdigit() and int(b)<=5 and int(b)>0:
            break
        else:
            print("You answerd wrong case => " + b)
            print("Please answer the question again.")
    
    while True:
        print("Question 3. 이 글은 사람이 쓴 것 같다.")
        print("1. 매우 그렇다. 2. 그렇다. 3. 보통이다. 4. 아니다. 5. 매우 아니다.")
        print("Answer the number.",end=" ")
        c=input()
        if c.isdigit() and int(c)<=5 and int(c)>0:
            break
        else:
            print("You answerd wrong case => " + c)
            print("Please answer the question again.")
    
    if sample["label"]=="fake":
        f_scores_a.append(int(a))
        f_scores_b.append(int(b))
        f_scores_c.append(int(c))
    else:
        r_scores_a.append(int(a))
        r_scores_b.append(int(b))
        r_scores_c.append(int(c))
    if debug:
        print("a : " + a)
        print("b : " + b)
        print("c : " + c)
    print("If you want to stop the survey, please enter the '0'. If you enter whatever else, the survey is keep going.")
    stop=input()
    print("You entered " + stop)
    if stop=='0':
        break





