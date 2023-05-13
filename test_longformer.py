import torch
import pickle
from tqdm import tqdm, trange
from dataset_consts import *
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig,LongformerModel
import sys

print("gpu : ")
print(torch.cuda.is_available())

testfile_name=sys.argv[1] # 예제 : wp_all_generations_outputs
save_dir=sys.argv[2] #all.tar
log_dir=sys.argv[3] # coh1
gpu=sys.argv[4] # cuda:0 or cpu
debug=int(sys.argv[5]) # 1 or 0

if debug==1:
    debug=True
else:
    debug=False

print("test file : " + testfile_name + ".csv")
print("save dir : " + save_dir)
print("log dir : " + log_dir)
print("gpu or cpu : " + gpu)
print("debug mode : " + str(debug))



CONTINUOUSLY_TRAIN=True


createFolder('longformer')
PATH = './longformer/'+save_dir

writer = SummaryWriter('./runs/'+log_dir)

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

class MyLongformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=AutoConfig.from_pretrained('allenai/longformer-base-4096')
        self.bert = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        self.rogistic=torch.nn.Linear(self.config.hidden_size,1)
        self.sigmoid=torch.nn.Sigmoid()
        self.loss=torch.nn.BCELoss()

    def forward(self, input_ids,attention_mask,global_attention_mask,labels=None):
        output=self.bert(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        prob=self.rogistic(output.pooler_output)
        prob=self.sigmoid(prob)
        loss=0
        if labels is not None:
            loss=self.loss(prob,labels)
        return prob, loss

# outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
# print(outputs.pooler_output.shape)
from transformers import get_scheduler
import torch.optim as optim
mylongformer=MyLongformer()
# print(mylongformer(input_ids,attention_mask,global_attention_mask,label=torch.FloatTensor([[1]])))
if torch.cuda.is_available():
     mylongformer=mylongformer.to(gpu)

if CONTINUOUSLY_TRAIN:
    checkpoint= torch.load(PATH)
    mylongformer.load_state_dict(checkpoint['model_state_dict'])

mylongformer.eval()
def eval(fake_outputs,real_outputs):
    fake=tokenizer(fake_outputs,max_length=4096,padding="max_length",
                truncation=True,return_tensors="pt")
    input_ids=fake['input_ids']
    attention_mask=fake['attention_mask']
    global_attention_mask=torch.zeros_like(attention_mask)
    global_attention_mask[:,0]=1
    fake_probs,_=mylongformer(input_ids=input_ids,attention_mask=attention_mask,global_attention_mask=global_attention_mask,)

    real=tokenizer(real_outputs,max_length=4096,padding="max_length",
                truncation=True,return_tensors="pt")
    input_ids=real['input_ids']
    attention_mask=real['attention_mask']
    global_attention_mask=torch.zeros_like(attention_mask)
    global_attention_mask[:,0]=1
    real_probs,_=mylongformer(input_ids=input_ids,attention_mask=attention_mask,global_attention_mask=global_attention_mask,)
    return fake_probs, real_probs

import csv
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

f = open(testfile_name+'.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
first=True

count=0
last_keywords=""
cumul_fake_outputs=""
cumul_real_outputs=""
f_score=0
r_score=0
f_scores=[]
r_scores=[]
for line in rdr:
    
    if first:
        first=False
        continue
    count+=1

    if debug:
        print("keywords : " + line[2][0])
        print("fake outputs : " + line[3][0])
        print("real outputs : " + line[4][0])
        input()

    if line[2][0]==last_keywords:
        cumul_fake_outputs+=line[4][0]
        cumul_real_outputs+=line[3][0]
        continue
    else:
        if count!=1:
            f_score,r_score=eval(cumul_fake_outputs,cumul_real_outputs)
            f_scores.append(f_score)
            r_scores.append(r_score)

            if debug:
                print("eval results : " )
                print(cumul_fake_outputs)
                print(cumul_real_outputs)
                print(last_keywords)
                print(f_score)
                print(r_score)
                print("###############")
            
        cumul_fake_outputs=line[4][0]
        cumul_real_outputs=line[3][0]
        last_keywords=line[2][0]


f_score,r_score=eval(cumul_fake_outputs,cumul_real_outputs)
f_scores.append(f_score)
r_scores.append(r_score)

f_scores=np.array(f_scores)
r_scores=np.array(r_scores)

print(testfile_name + "'s " + save_dir + " mean score : " + str(np.mean(f_scores)) + "\n var : " + str(np.var(f_scores)))
print("and this is baseline (original dataset)'s same mean score : " + str(np.mean(r_scores))+ "\n var : " + str(np.var(r_scores)))
