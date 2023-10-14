import sys
import torch
from tqdm import tqdm, trange
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
import csv
import ctypes as ct
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import pickle
import random
from dataset_consts import *
import gc
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
num_added_toks = tokenizer.add_tokens(["[SEP]"],special_tokens=True)
t_v_t=sys.argv[1]
dataset_name=sys.argv[2] # 예제 : coherence (저장될 이름)
dataset_dir=sys.argv[3] # 예제 : whole

bart_tokenizer=AutoTokenizer.from_pretrained("facebook/bart-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
if "nextsentenceprediction" in dataset_name:
    tokenizer.add_tokens(["[SEP]"],special_tokens=True)


def get_real_train_data():
    whole_new_dataset=[]
    whole_new_dataset_length=[]
    batch_size=1
    new_whole_data=[]
    for i in range(50):
        new_whole_data.append([])
    for j in trange(1,50): # 최대 100개 문단까지 있다.
    
        if dataset_dir !="whole":
            
            with open("pickle_data/"+"bart_" + str(t_v_t) + "_" + dataset_dir+"/level_2_" + str(j) + ".pickle","rb") as fi:
                    train_dataset = pickle.load(fi)
        else: # whole dataset train.
            with open("pickle_data/"+"bart_" + str(t_v_t) +"_"+ "wp_rake"+"/level_2_" + str(j) + ".pickle","rb") as fi:
                    train_dataset = pickle.load(fi)
            
            with open("pickle_data/"+"bart_"+ str(t_v_t) +"_"+"booksum_rake"+"/level_2_" + str(j) + ".pickle","rb") as fi:
                    train_dataset += pickle.load(fi)
            
            with open("pickle_data/"+"bart_"+ str(t_v_t) +"_"+"reedsy_rake"+"/level_2_" + str(j) + ".pickle","rb") as fi:
                    train_dataset += pickle.load(fi)   
        if len(train_dataset)==0:
            continue
        
        whole_new_dataset_length.append(len(train_dataset))
        
        # print("the training set for " + str(i) + " Num Paragramphs.")
        for i in range(0, len(train_dataset),batch_size):
            # get the inputs; data is a list of [inputs, labels]
                if i+batch_size>len(train_dataset):
                    # batch size에 안 맞는 마지막 set은 , 그냥 버린다
                    # batch size는 커봐야 4 정도니까 이정도는 괜찮다.
                    
                    break

                
                batch_data=train_dataset[i:i+batch_size]
                
                first=True
                batch_num_decoder_input_ids=[]
                batch_decoder_attention_masks=[]
                for data in batch_data:
                    input_ids,attention_mask,num_decoder_input_ids,decoder_attention_masks,prompt= (data['input_ids'],data['input_attention'],data['decoder_input_ids'],data['decoder_attention_mask'],data['prompt'])
                    batch_num_decoder_input_ids.append(bart_tokenizer.batch_decode(num_decoder_input_ids,skip_special_tokens=True))# 또 각각 decoder_input_ids에서도 <s>는 떼내야 한다.
                    #이건 데이터셋 만들때 처리해줘야 할듯하다.
                    #batch_decoder_attention_masks.append(decoder_attention_masks)
                    """
                    if first:
                        batch_input_ids=input_ids # 생각해보니 input_ids에서 </s>를 떼야 될 것 같다.
                        batch_attention_mask=attention_mask
                        batch_prev_predictions=prompt
                        first=False
                    else:
                        batch_input_ids=torch.cat((batch_input_ids,input_ids),dim=0)
                        batch_attention_mask=torch.cat((batch_attention_mask,attention_mask),dim=0)
                        
                        batch_prev_predictions=torch.cat((batch_prev_predictions,prompt),dim=0)
                    """
                #batch_num_decoder_input_ids=torch.stack(batch_num_decoder_input_ids,dim=1)
                #batch_decoder_attention_masks=torch.stack(batch_decoder_attention_masks,dim=1)

                # whole_new_dataset.append({'batch_input_ids':batch_input_ids,'batch_attention_mask':batch_attention_mask,'batch_prev_predictions':batch_prev_predictions,
                #                         'batch_num_decoder_input_ids' : batch_num_decoder_input_ids,'batch_decoder_attention_mass':batch_decoder_attention_masks})
                #print(batch_num_decoder_input_ids)
                #input()
                if len(batch_num_decoder_input_ids)!=0:
                    #print(batch_num_decoder_input_ids[0])
                    new_whole_data[j].append(batch_num_decoder_input_ids[0])

    return new_whole_data


def get_whole_data(wp=False,reedsy=False,booksum=False,t_v_t="train",location="../writingPrompts/",start=0,range=0):

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
    for i in range(50):
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
                now_sentences+=sentences+" "
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
        """
        for s in split_s: 
            print(s)
            print("---------")
        """
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
    """
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
            print(sample[i])
            sample[i][-1]=sample[i+1][-1]

            whole_data_1=' '.join(sample[i])
            neg_examples_2.append({'data' : whole_data_1,'label':[0]})
            print("index : " + str(i) + " whole_data_1 : " + whole_data_1)
            input()

        for j in range(len(sample)//2,len(sample)):
            whole_data_3=' '.join(sample[j])
            pos_examples_2.append({'data' : whole_data_3,'label':[1]})
    """
    list_of_ending=["Thanks for reading","Thank you for reading", "https", "Prompts", "Prompt", "Writing", "Tweeter"]
    for i in range(0,len(new_whole_data[1])):
        
        sentences=sent_tokenize(''.join(new_whole_data[1][i]))
        temp_sentences=copy.deepcopy(sentences)
        for sentence in sentences:
            remove=False
            for ending in list_of_ending:
                if ending in sentence:
                    remove=True
                    break
            if remove is True:
                temp_sentences.remove(sentence)
        
        whole_data_2=' '.join(temp_sentences)
        pos_examples_2.append({'data' : whole_data_2,'label':[1]})

    for num, sample in enumerate(new_whole_data[2:]):
        for i in range(0,len(sample),5):
            
            for sample_parag_num in range(0,len(sample[i])):
                #print(sample[i][sample_parag_num])
                sentences=sent_tokenize(''.join(sample[i][sample_parag_num]))
                temp_sentences=copy.deepcopy(sentences)
                for sentence in sentences:
                    remove=False
                    for ending in list_of_ending:
                        if ending in sentence:
                            remove=True
                            break
                    if remove is True:
                        temp_sentences.remove(sentence)
                
                sample[i][sample_parag_num]=' '.join(temp_sentences)
                #print(sample[i][sample_parag_num])

            #print("ending job done.")
            #input()
            for sample_parag_num in range(0,len(sample[i])-1):
                #print(sample[i][sample_parag_num])
                
                # neg_sample=random.choice(sample[i][:-1])
                
                neg_sample=sample[i][sample_parag_num]
                neg_sample=neg_sample.replace('\n',' ').replace('\\',' ')
                neg_examples_2.append({'data' : neg_sample,'label':[0]})
            #print("index : " + str(i) + " whole_data_1 : " + neg_sample)
            #input()
            pos_sample=sample[i][-1]
            pos_sample=pos_sample.replace('\n',' ').replace('\\',' ')
            for sample_parag_num in range(0,len(sample[i])-1):
                pos_examples_2.append({'data' : pos_sample,'label':[1]})

                
    

    print("whole pos length : " + str(len(pos_examples_2)))
    print(pos_examples_2[-1])
    print("whole neg length : " + str(len(neg_examples_2)))
    print(neg_examples_2[-1])
    examples_2=neg_examples_2+pos_examples_2
    print("whole length : " + str(len(examples_2)))


    return examples_2

def making_nextsentenceprediction_examples(new_whole_data):
    examples_3=[]
    neg_examples_3=[]
    pos_examples_3=[]


    for num, sample in enumerate(new_whole_data[2:]):
        for i in range(0,len(sample),5):
            for sample_parag_num in range(0,len(sample[i])):
                # sample_parag_num=random.randint(1,len(sample[i])-1) # 예를들어 길이가 3이면, 1~2까지 랜덤한 문단 하나를 뽑는다. (마지막 문단 포함)
                neg_sample=sample[i][sample_parag_num]
                if sample_parag_num==0:
                    random_sample_parag_num=random.choice(list(range(0,sample_parag_num+1))+list(range(sample_parag_num+2,len(sample[i]))))
                else:
                    random_sample_parag_num=random.choice([sample_parag_num-1,sample_parag_num])
                neg_sample += "[SEP]" + " " + sample[i][random_sample_parag_num]  # 이렇게 하면 원래 정상적으로 뒤에 올 문단을 제외한 모든 문단이 뒤에 붙을 가능성이 생긴다.
                """
                second_sample_parag_num=random.randint(0,len(sample[i])-1) # 같은 샘플에서 또 랜덤 하나를 뽑는다.
                
                if (sample_parag_num+1)==second_sample_parag_num: #만약 정상적인 경우(즉 0-1 혹은 1-2 이렇게 뽑힌 경우)
                    temp=second_sample_parag_num
                    second_sample_parag_num=sample_parag_num
                    sample_parag_num=temp #(둘의 순서를 바꾼다. 1-0 이렇게.)
                """
                neg_sample=neg_sample.replace('\n',' ').replace('\\',' ')
                neg_examples_3.append({'data' : neg_sample,'label':[0]})
                print("index : " + str(i) + " whole_data_1 : " + neg_sample)
                input()
        for j in range(0,len(sample),5):
            pos_samples=[]
            for sample_parag_num in range(0,len(sample[j])-1):
                # sample_parag_num=random.randint(0,len(sample[j])-2) # 예를들어 길이가 3이면, 0~1까지 랜덤한 문단 하나를 뽑는다. (마지막 문단만 빼고.)
                pos_sample=sample[j][sample_parag_num]
                pos_sample += "[SEP]" + " " + sample[j][sample_parag_num+1]
                pos_sample=pos_sample.replace('\n',' ').replace('\\',' ')
                pos_samples.append(pos_sample)
                pos_examples_3.append({'data' : pos_sample,'label':[1]})
            pos_examples_3.append({'data' : random.choice(pos_samples),'label':[1]}) # neg sample과 개수를 맞춰주기 위해서, 일부러 pos sample 중 랜덤하게 하나를 골라서 추가해준다.        

    print("whole pos length : " + str(len(pos_examples_3)))
    print(pos_examples_3[-2])
    
    print("whole neg length : " + str(len(neg_examples_3)))
    print(neg_examples_3[-2])
    examples_3=neg_examples_3+pos_examples_3
    print("whole length : " + str(len(examples_3)))


    return examples_3
# examples=pos_examples+neg_examples
# # 원하는 꼴.
# # pos_examples => [{'data' : [[CLS] + '~~' + [SEP] + '~~'], 'label' : [1]}, {}, {},...]
# # neg_examples => [{'data' : [[CLS] + '~~' + [SEP] + '~~'], 'label' : [0]}, {}, {},...]
# # examples => pos_examples + neg_examples, 그리고 random.shuffle(examples)
def making_logical_examples(new_whole_data):
    examples_4=[]
    neg_examples_4=[]
    pos_examples_4=[]
    

    for num, sample in enumerate(new_whole_data[2:]):
        for i in range(0,len(sample)//2):
            
            if sample[i]=='':
                continue

            
            #print(sample[i])

            sentences=sent_tokenize(' '.join(sample[i]))
            #print(sentences)
            
            #print()
            #print(' '.join(sentences))

            random.shuffle(sentences)
            #print(sentences)
            if len(sentences)==0:
                continue

            
            
            neg_sample=' '.join(sentences)
            
            #print(neg_sample)
            #input()
            neg_examples_4.append({'data' : neg_sample,'label':[0]})
            #print("index : " + str(i) + " whole_data_1 : " + neg_sample)
            #input()

        for j in range(len(sample)//2,len(sample)):
            
            
            ct=' '.join(sent_tokenize(' '.join(sample[j])))
            
            #print(ct)
            #input()

            if len(ct)==0:
                continue
            pos_sample=ct
            pos_sample=pos_sample[1:]
            #print(pos_sample)
            #input()
            pos_examples_4.append({'data' : pos_sample,'label':[1]})
    

    print("whole pos length : " + str(len(pos_examples_4)))
    print("whole neg length : " + str(len(neg_examples_4)))
    examples_4=neg_examples_4+pos_examples_4
    print("whole length : " + str(len(examples_4)))


    return examples_4

def making_pickle_data(examples,name):
    df=pd.DataFrame(examples)
    labels=torch.FloatTensor(df['label'].values.tolist())
    datas=df['data'].values.tolist()

    token_datas=tokenizer(datas,max_length=500,padding="max_length",
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

t_v_t=t_v_t
examples_1=[]
examples_2=[]

examples=[]
new_whole_data=get_real_train_data()
print("load done. dataset maker start.")
print(new_whole_data[1][0])
if "completeness" in dataset_name:
    examples=making_completeness_examples(new_whole_data)
    making_pickle_data(examples,"coherence_completeness/"+t_v_t+"_"+dataset_name)
elif "nextsentenceprediction" in dataset_name:
    examples=making_nextsentenceprediction_examples(new_whole_data)
    making_pickle_data(examples,"coherence_completeness/"+t_v_t+"_"+dataset_name)
else:
    print("there is no name of examples : " + dataset_name)


# if "wp" in dataset_name: 
#     whole_data=get_whole_data(wp=True,t_v_t=t_v_t,start=start,range=100000)
#     new_whole_data=making_new_whole_data(whole_data) # 문단별로 자름.
#     del whole_data
#     #report(new_whole_data)
#     #wp_examples_1=making_coherence_examples(new_whole_data)
#     #wp_examples_2=making_completeness_examples(new_whole_data)
#     #wp_examples_3=making_nextsentenceprediction_examples(new_whole_data)
#     examples+=making_logical_examples(new_whole_data)
#     del new_whole_data
#     gc.collect()

# if "rd" in dataset_name:
#     whole_data=get_whole_data(reedsy=True,t_v_t=t_v_t,start=0,range=0)
#     new_whole_data=making_new_whole_data(whole_data) # 문단별로 자름.
#     del whole_data
#     #report(new_whole_data)
#     #rd_examples_1=making_coherence_examples(new_whole_data)
#     #rd_examples_2=making_completeness_examples(new_whole_data)
#     #rd_examples_3=making_nextsentenceprediction_examples(new_whole_data)
#     examples+=making_logical_examples(new_whole_data)

# if "bk" in dataset_name:
#     whole_data=get_whole_data(booksum=True,location="../booksum/",t_v_t=t_v_t,start=0,range=0)
#     new_whole_data=making_new_whole_data(whole_data) # 문단별로 자름.
#     del whole_data
#     #report(new_whole_data)
#     #bk_examples_1=making_coherence_examples(new_whole_data)
#     #bk_examples_2=making_completeness_examples(new_whole_data)
#     #bk_examples_3=making_nextsentenceprediction_examples(new_whole_data)
#     examples+=making_logical_examples(new_whole_data)

#     del new_whole_data
#     gc.collect()


#examples_1=wp_examples_1+bk_examples_1+rd_examples_1
#examples_2=wp_examples_2+bk_examples_2+rd_examples_2


#examples_3=wp_examples_3+bk_examples_3+rd_examples_3

#examples_4=wp_examples_4+rd_examples_4+bk_examples_4


#making_pickle_data(examples_1,"coherence_completeness/"+t_v_t+"_coherence-1")
#del examples_1
#gc.collect()
#making_pickle_data(examples_2,"coherence_completeness/"+t_v_t+"_completeness-1")
#making_pickle_data(examples_3,"coherence_completeness/"+t_v_t+"_nextsentenceprediction-5")
