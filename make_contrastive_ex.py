import torch
from tqdm import tqdm, trange
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large")
import csv
import ctypes as ct
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import pickle
import random
from dataset_consts import *
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))





def get_whole_data(wp=False,reedsy=False,booksum=False,t_v_t="train",location="./writingPrompts/",start=0,range=0):

    if wp:
        T=t_v_t

        total_source=[]       
        with open(location+ T +".wp_source", encoding='UTF8') as f:
            stories = f.readlines()
            stories = [" ".join(i.split()[0:1000]) for i in stories]
            temp_stories=[]
            for story in stories:
                temp_stories.append(story.replace("<newline>",""))
            total_source.append(temp_stories)

        total_target=[]

        with open(location+ T +".wp_target", encoding='UTF8') as f:
            stories = f.readlines()
            stories = [" ".join(i.split()[0:1000]) for i in stories]
            temp_stories=[]
            for story in stories:
                temp_stories.append(story.replace("<newline>",""))
            total_target.append(temp_stories)

        RANGE=range
        START=start # 마지막으로 끝난 라인부터 다시 만든다.

        if RANGE !=0:
            whole_data=total_target[0][START:START+RANGE]
        else:
            whole_data=total_target[0][START:]
    
    elif booksum:
        if t_v_t=="valid":
            t_v_t="dev"
        file=location+t_v_t
        f = pd.read_csv(file+'.csv',chunksize=1000)
        f= pd.concat(f)
        first=True
        text=[]
        for index, line in f.iterrows():
            if first:
                first=False
                continue
            chapter=line[9]
            text.append(chapter)
        
        whole_data=text
        RANGE=range
        START=start
        if RANGE !=0:
            whole_data=whole_data[START:START+RANGE]
        else:
            whole_data=whole_data[START:]
        
    elif reedsy:
        whole_data=[]
        with open(t_v_t+"_reedsy_prompts_whole.pickle","rb") as fi:
            reedsy = pickle.load(fi)
        for line in reedsy:
            whole_data.append(line['story'])
        
        RANGE=range
        START=start
        if RANGE !=0:
            whole_data=whole_data[START:START+RANGE]
        else:
            whole_data=whole_data[START:]
        
    return whole_data # it should be (N, string)



def making_new_whole_data(whole_data):

    new_whole_data=[]
    for i in range(100):
        new_whole_data.append([])


    for t in tqdm(whole_data):
        
        tt=sent_tokenize(t)

        # print(tt)
        # print(len(tt))

        prev=0
        
        split_s=[]
        now_sentences=""
        
        for sentences in tt:
            t_s=tokenizer(sentences).input_ids
            if len(t_s)+prev<=200:
                now_sentences+=" " + sentences
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
        if len(split_s)>=len(new_whole_data):
            continue
        new_whole_data[len(split_s)].append(split_s)
    
    return new_whole_data # it should be (100,M,K) --> M은 각각 문단 개수별 예제의 개수, K는 문단 개수.

def report(new_whole_data):
    for i,sample in enumerate(new_whole_data[1:],start=1):
        if len(sample)==0:
            continue
        print("문단"+ str(i) + " 개 짜리 개수 : " + str(len(sample)))
        avg_len=0
        for split_s in sample:
            for one in split_s:
                avg_len+=len(tokenizer(one).input_ids)
        avg_len=avg_len/(i*len(sample))
        print("각 문단 평균 토큰의 개수 : "+ str(avg_len))
        
def making_coherence_examples(new_whole_data):
    examples=[]
    neg_examples=[]
    pos_examples=[]
    for i in range(0,len(new_whole_data[1])//2,2):
        if i+1>=len(new_whole_data[1])//2:
            whole_data_3=' '.join(new_whole_data[1][i])
            pos_examples.append({'data' : whole_data_3,'label':[1]})
            break


        sentences_1=sent_tokenize(' '.join(new_whole_data[1][i]))
        sentences_2=sent_tokenize(' '.join(new_whole_data[1][i+1]))
        length=min(len(sentences_1),len(sentences_2))
        
        # print(sentences_1)
        # print(sentences_2)
        
        if length==1: # 단 하나짜리 문장은 거의 없겠지만, 혹시 있다면 걍 pos example로 넣자.
            whole_data_3=' '.join(new_whole_data[1][i])
            pos_examples.append({'data' : whole_data_3,'label':[1]})
            
            # print("index : " + str(i) + " whole_data_3 : " + whole_data_3)

            whole_data_3=' '.join(new_whole_data[1][i+1])
            pos_examples.append({'data' : whole_data_3,'label':[1]})

            
            # print("index : " + str(i+1) + " whole_data_3 : " + whole_data_3)
            # input()

            continue
        
        random_sentences_1=random.sample(range(0,len(sentences_1)),length//2)
        random_sentences_2=random.sample(range(0,len(sentences_2)),length//2)

        for l in range(length//2):
            temp=sentences_2[random_sentences_2[l]]  
            sentences_2[random_sentences_2[l]]=sentences_1[random_sentences_1[l]]
            sentences_1[random_sentences_1[l]]=temp
        
        whole_data_1=' '.join(sentences_1)
        whole_data_2=' '.join(sentences_2)
        neg_examples.append({'data' : whole_data_1,'label':[0]})
        neg_examples.append({'data' : whole_data_2,'label':[0]})
        
        # print("index : " + str(i) + " whole_data_1 : " + whole_data_1 + "\nwhole_data_2 : " + whole_data_2)
        # input()

    for j in range(len(new_whole_data[1])//2,len(new_whole_data[1])):
            pos_examples.append({'data' : ' '.join(new_whole_data[1][j]),'label':[1]})
            # print("index : " + str(j) + " whole_data_3 : " + ' '.join(new_whole_data[1][j]))


    for num, sample in enumerate(new_whole_data[2:],start=1):
        for i in range(0,len(sample)//2,2):
            if i+1>=len(sample)//2:
                whole_data_3=' '.join(sample[i])
                pos_examples.append({'data' : whole_data_3,'label':[1]})
                break

            # print(sample[i])
            # print(sample[i+1])

            random_paragraph=random.randint(0,num)
            temp=sample[i+1][random_paragraph]  
            sample[i+1][random_paragraph]=sample[i][random_paragraph]
            sample[i][random_paragraph]=temp
            whole_data_1=' '.join(sample[i])
            whole_data_2=' '.join(sample[i+1])
            neg_examples.append({'data' : whole_data_1,'label':[0]})
            neg_examples.append({'data' : whole_data_2,'label':[0]})

            # print("index : " + str(i) + " whole_data_1 : " + whole_data_1 + "\nwhole_data_2 : " + whole_data_2)
            # input()

        for j in range(len(sample)//2,len(sample)):
            whole_data_3=' '.join(sample[j])
            pos_examples.append({'data' : whole_data_3,'label':[1]})

    print("whole pos length : " + str(len(pos_examples)))
    print("whole neg length : " + str(len(neg_examples)))
    examples=neg_examples+pos_examples
    print("whole length : " + str(len(examples)))
    return examples
# print(neg_examples[1])
# print(neg_examples[400])


def making_completeness_examples(new_whole_data):
    examples_2=[]
    neg_examples_2=[]
    pos_examples_2=[]
    for i in range(0,len(new_whole_data[1])//2):
        sentences=sent_tokenize(' '.join(new_whole_data[1][i]))
        # print(sentences)
        if (len(sentences)//2>=1):
            sentences=sentences[:len(sentences)-len(sentences)//2]
            whole_data_1=' '.join(sentences)
            neg_examples_2.append({'data' : whole_data_1,'label':[0]})
            # print("index : " + str(i) + " whole_data_1 : " + whole_data_1)
            # input()
        else: # 역시 한 문장짜리는 그냥 pos example로 넘긴다.
            whole_data_2=' '.join(sentences)
            pos_examples_2.append({'data' : whole_data_2,'label':[1]})

    for j in range(len(new_whole_data[1])//2, len(new_whole_data[1])):
        pos_examples_2.append({'data' : ' '.join(new_whole_data[1][j]),'label':[1]})


    for num, sample in enumerate(new_whole_data[2:]):
        for i in range(0,len(sample)//2):
            # print(sample[i])
            sample[i].pop(-1) # 마지막 요소를 날린다.
            whole_data_1=' '.join(sample[i])
            neg_examples_2.append({'data' : whole_data_1,'label':[0]})
            # print("index : " + str(i) + " whole_data_1 : " + whole_data_1)
            # input()

        for j in range(len(sample)//2,len(sample)):
            whole_data_3=' '.join(sample[j])
            pos_examples_2.append({'data' : whole_data_3,'label':[1]})

    print("whole pos length : " + str(len(pos_examples_2)))
    print("whole neg length : " + str(len(neg_examples_2)))
    examples_2=neg_examples_2+pos_examples_2
    print("whole length : " + str(len(examples_2)))
    return examples_2

# examples=pos_examples+neg_examples
# # 원하는 꼴.
# # pos_examples => [{'data' : [[CLS] + '~~' + [SEP] + '~~'], 'label' : [1]}, {}, {},...]
# # neg_examples => [{'data' : [[CLS] + '~~' + [SEP] + '~~'], 'label' : [0]}, {}, {},...]
# # examples => pos_examples + neg_examples, 그리고 random.shuffle(examples)


def making_pickle_data(examples,name):
    df=pd.DataFrame(examples)
    labels=torch.FloatTensor(df['label'].values.tolist())
    datas=df['data'].values.tolist()
    token_datas=tokenizer(datas,max_length=4096,padding="max_length",
                truncation=True,return_tensors="pt")
    input_ids=token_datas['input_ids']
    attention_mask=token_datas['attention_mask']
    global_attention_mask=torch.zeros_like(attention_mask)
    global_attention_mask[:,0]=1
    print(input_ids.shape)
    print(attention_mask.shape)
    print(global_attention_mask.shape)
    print(labels.shape)


    train_dataset=Contrastive_Dataset(input_ids,attention_mask,global_attention_mask,labels)
    with open(name+".pickle","wb") as f:
        pickle.dump(train_dataset,f)

t_v_t="valid"

whole_data=get_whole_data(wp=True,t_v_t=t_v_t,start=0,range=1000)
new_whole_data=making_new_whole_data(whole_data) # 문단별로 자름.
report(new_whole_data)
wp_examples_1=making_coherence_examples(new_whole_data)
wp_examples_2=making_completeness_examples(new_whole_data)

whole_data=get_whole_data(reedsy=True,t_v_t=t_v_t,start=0,range=1000)
new_whole_data=making_new_whole_data(whole_data) # 문단별로 자름.
report(new_whole_data)
rd_examples_1=making_coherence_examples(new_whole_data)
rd_examples_2=making_completeness_examples(new_whole_data)

whole_data=get_whole_data(booksum=True,location="booksum/",t_v_t=t_v_t,start=0,range=1000)
new_whole_data=making_new_whole_data(whole_data) # 문단별로 자름.
report(new_whole_data)
bk_examples_1=making_coherence_examples(new_whole_data)
bk_examples_2=making_completeness_examples(new_whole_data)

examples_1=wp_examples_1+bk_examples_1+rd_examples_1
examples_2=wp_examples_2+bk_examples_2+rd_examples_2

making_pickle_data(examples_1,t_v_t+"_coherence")
making_pickle_data(examples_2,t_v_t+"_completeness")

