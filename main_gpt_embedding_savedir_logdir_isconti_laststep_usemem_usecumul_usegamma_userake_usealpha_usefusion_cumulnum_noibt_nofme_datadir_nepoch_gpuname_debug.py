

import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForCausalLM
from transformers import AutoConfig
from dataset_consts_gpt import *
import random
from torch.utils.tensorboard import SummaryWriter
from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer
from rake_nltk import Rake
r = Rake()
import evaluate
import sys

metric = evaluate.load("rouge")
meteor=evaluate.load("meteor")
_bleu=evaluate.load("bleu")


def sorting(lst):
    # lst2=sorted(lst, key=len)
    lst2 = sorted(lst, key=len)
    return lst2
def clean_top_features(keywords, top=10):
    keywords = sorting(keywords)
    newkeys = []
    newkeys.append(keywords[len(keywords)-1])
    for i in range(len(keywords)-2,-1,-1):
        if newkeys[len(newkeys)-1].startswith(keywords[i]):
            continue
        newkeys.append(keywords[i])

    if len(newkeys) > top:
        return newkeys[:top]
    return newkeys
def convert_keys_to_str(key_list):
    newstr = key_list[0]
    for k in range(1, len(key_list)):
        if len(key_list[k].split(' ')) > 2 :
            newstr += '[SEP]' + key_list[k]
    return newstr.replace("(M)", "").strip()
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error Creating directory. ' + directory)

LAST_PARAG=5

save_dir=sys.argv[1] # rake_all
log_dir=sys.argv[2] # rake_all

conti=int(sys.argv[3]) # 0 

last_step=int(sys.argv[4]) # 0
use_mem=int(sys.argv[5]) # 1
use_cumul=int(sys.argv[6]) # 1
use_gamma=int(sys.argv[7])
use_rake=int(sys.argv[8])
use_alpha=int(sys.argv[9])
use_fusion=int(sys.argv[10])

cumul_num=int(sys.argv[11]) # 3

no_ibt=int(sys.argv[12])
no_fme=int(sys.argv[13])

dataset_dir=sys.argv[14]
num_epochs=int(sys.argv[15])

gpu_name=sys.argv[16] # cuda:0
debug = int(sys.argv[17]) # 1
if debug ==1:
    debug=True
else:
    debug=False

createFolder('second_level')
PATH = './second_level/'+save_dir+'.tar'
writer = SummaryWriter("./runs/"+log_dir)
CONTINUOUSLY_TRAIN=False
if conti==1:
    CONTINUOUSLY_TRAIN=True
LAST_STEP=last_step

USE_MEMORY=True
if use_mem==1:
    USE_MEMORY=True
else:
    USE_MEMORY=False

USE_CUMULATIVE=True
if use_cumul==1:
    USE_CUMULATIVE=True
else:
    USE_CUMULATIVE=False
USE_GAMMA=True
if use_gamma==1:
    USE_GAMMA=True
else:
    USE_GAMMA=False
USE_RAKE=True
if use_rake==1:
    USE_RAKE=True
else:
    USE_RAKE=False
USE_FUSION=True
if use_fusion==1:
    USE_FUSION=True
else:
    USE_FUSION=False

USE_ALPHA=True
if use_alpha==1:
    USE_ALPHA=True
else:
    USE_ALPHA=False

TEACHER_FORCING_MEMORY=True
CUMUL_NUM=cumul_num
NO_IBT=False
if no_ibt==1:
    NO_IBT=True
else:
    NO_IBT=False
NO_FME=False
if no_fme==1:
    NO_FME=True
else:
    NO_FME=False


print('save_dir : ' + save_dir)
print('log_dir : ' + log_dir)
print('use_cumulative : ')
print(USE_CUMULATIVE)
print('cumul num :')
print(cumul_num)
print('use_mem : ')
print(USE_MEMORY)
print('use_rake : ')
print(USE_RAKE)
print('use alpha :')
print(USE_ALPHA)
print('use fusion :')
print(USE_FUSION)
print('continuously : ')
print(CONTINUOUSLY_TRAIN)
print("last_step : ")
print(last_step)
print("no_ibt : ")
print(NO_IBT)
print("no_fme : ")
print(NO_FME)

print("dataset dir : ")
print(dataset_dir)
print("num_epochs:")
print(num_epochs)
print("gpu or cpu num :")
print(gpu_name)
print("debug : ")
print(debug)
# num_added_toks = tokenizer.add_tokens(["<plot>","</plot>","<prev>","</prev>","<by>","<sep>"],special_tokens=True)
num_added_toks = tokenizer.add_tokens(["<plot>","</plot>","<prev>","</prev>","<i>","<b>","<t>","<f>","<m>","<e>","[SEP]","<n_e>"],special_tokens=True)
soplot_id=tokenizer.convert_tokens_to_ids("<plot>")
eoplot_id=tokenizer.convert_tokens_to_ids("</plot>")
soprev_id=tokenizer.convert_tokens_to_ids("<prev>")
eoprev_id=tokenizer.convert_tokens_to_ids("</prev>")
sep_id=tokenizer.convert_tokens_to_ids("[SEP]")
intro_id=tokenizer.convert_tokens_to_ids("<i>")
body_id=tokenizer.convert_tokens_to_ids("<b>")
tail_id=tokenizer.convert_tokens_to_ids("<t>")
front_id=tokenizer.convert_tokens_to_ids("<f>")
middle_id=tokenizer.convert_tokens_to_ids("<m>")
ending_id=tokenizer.convert_tokens_to_ids("<e>")
next_is_ending_id=tokenizer.convert_tokens_to_ids("<n_e>")

# by_id=tokenizer.convert_tokens_to_ids("<by>")
soplot_token_tensor=torch.LongTensor([[soplot_id]]).to(gpu_name)
eoplot_token_tensor=torch.LongTensor([[eoplot_id]]).to(gpu_name)
soprev_token_tensor=torch.LongTensor([[soprev_id]]).to(gpu_name)
eoprev_token_tensor=torch.LongTensor([[eoprev_id]]).to(gpu_name)
sep_token_tensor=torch.LongTensor([[sep_id]]).to(gpu_name)
# by_token_tensor=torch.LongTensor([[by_id]]).to(gpu_name)
intro_token_tensor=torch.LongTensor([[intro_id]]).to(gpu_name)
body_token_tensor=torch.LongTensor([[body_id]]).to(gpu_name)
tail_token_tensor=torch.LongTensor([[tail_id]]).to(gpu_name)
front_token_tensor=torch.LongTensor([[front_id]]).to(gpu_name)
middle_token_tensor=torch.LongTensor([[middle_id]]).to(gpu_name)
ending_token_tensor=torch.LongTensor([[ending_id]]).to(gpu_name)
next_is_ending_token_tensor=torch.LongTensor([[next_is_ending_id]]).to(gpu_name)

batch_sep_token_tensors=torch.cat([sep_token_tensor]*batch_size,dim=0) #(b,1)
batch_eoprev_token_tensors=torch.cat([eoprev_token_tensor]*batch_size,dim=0)
batch_soprev_token_tensors=torch.cat([soprev_token_tensor]*batch_size,dim=0) #(b,1)
batch_eoplot_token_tensors=torch.cat([eoplot_token_tensor]*batch_size,dim=0)
batch_soplot_token_tensors=torch.cat([soplot_token_tensor]*batch_size,dim=0) #(b,1)
batch_intro_token_tensors=torch.cat([intro_token_tensor]*batch_size,dim=0)
batch_body_token_tensors=torch.cat([body_token_tensor]*batch_size,dim=0) #(b,1)
batch_tail_token_tensors=torch.cat([tail_token_tensor]*batch_size,dim=0)
batch_front_token_tensors=torch.cat([front_token_tensor]*batch_size,dim=0) #(b,1)
batch_middle_token_tensors=torch.cat([middle_token_tensor]*batch_size,dim=0)
batch_ending_token_tensors=torch.cat([ending_token_tensor]*batch_size,dim=0) #(b,1)
batch_next_is_ending_token_tensors=torch.cat([next_is_ending_token_tensor]*batch_size,dim=0)



import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler,AutoTokenizer

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

num_added_toks = bert_tokenizer.add_tokens(["<plot>","</plot>","<prev>","</prev>","<i>","<b>","<t>","<f>","<m>","<e>","[SEP]","<n_e>"],special_tokens=True)
b_soprev_id=bert_tokenizer.convert_tokens_to_ids("<prev>")
b_eoprev_id=bert_tokenizer.convert_tokens_to_ids("</prev>")
b_sep_id=bert_tokenizer.convert_tokens_to_ids("[SEP]")
b_intro_id=bert_tokenizer.convert_tokens_to_ids("<i>")
b_body_id=bert_tokenizer.convert_tokens_to_ids("<b>")
b_tail_id=bert_tokenizer.convert_tokens_to_ids("<t>")
b_front_id=bert_tokenizer.convert_tokens_to_ids("<f>")
b_middle_id=bert_tokenizer.convert_tokens_to_ids("<m>")
b_ending_id=bert_tokenizer.convert_tokens_to_ids("<e>")
b_next_is_ending_id=bert_tokenizer.convert_tokens_to_ids("<n_e>")

b_soprev_token_tensor=torch.LongTensor([[b_soprev_id]]).to(gpu_name)
b_eoprev_token_tensor=torch.LongTensor([[b_eoprev_id]]).to(gpu_name)
b_sep_token_tensor=torch.LongTensor([[b_sep_id]]).to(gpu_name)
# by_token_tensor=torch.LongTensor([[by_id]]).to(gpu_name)
b_intro_token_tensor=torch.LongTensor([[b_intro_id]]).to(gpu_name)
b_body_token_tensor=torch.LongTensor([[b_body_id]]).to(gpu_name)
b_tail_token_tensor=torch.LongTensor([[b_tail_id]]).to(gpu_name)
b_front_token_tensor=torch.LongTensor([[b_front_id]]).to(gpu_name)
b_middle_token_tensor=torch.LongTensor([[b_middle_id]]).to(gpu_name)
b_ending_token_tensor=torch.LongTensor([[b_ending_id]]).to(gpu_name)
b_next_is_ending_token_tensor=torch.LongTensor([[b_next_is_ending_id]]).to(gpu_name)

batch_b_sep_token_tensors=torch.cat([sep_token_tensor]*batch_size,dim=0) #(b,1)
batch_b_eoprev_token_tensors=torch.cat([b_eoprev_token_tensor]*batch_size,dim=0)
batch_b_soprev_token_tensors=torch.cat([b_soprev_token_tensor]*batch_size,dim=0) #(b,1)
batch_b_intro_token_tensors=torch.cat([b_intro_token_tensor]*batch_size,dim=0)
batch_b_body_token_tensors=torch.cat([b_body_token_tensor]*batch_size,dim=0) #(b,1)
batch_b_tail_token_tensors=torch.cat([b_tail_token_tensor]*batch_size,dim=0)
batch_b_front_token_tensors=torch.cat([b_front_token_tensor]*batch_size,dim=0) #(b,1)
batch_b_middle_token_tensors=torch.cat([b_middle_token_tensor]*batch_size,dim=0)
batch_b_ending_token_tensors=torch.cat([b_ending_token_tensor]*batch_size,dim=0) #(b,1)
batch_b_next_is_ending_token_tensors=torch.cat([b_next_is_ending_token_tensor]*batch_size,dim=0)

class Network(nn.Module): 
   def __init__(self, shared, vocab_size, d_model,gpt,bert,bert_config): 
       super(Network, self).__init__() 
        
       self.shared = shared
       #nn.Embedding(config.vocab_size, config.d_model) 
        # 이 shared는 역할상 고정되어 있어야 한다.
       # 하지만 bart의 embedding layer는 학습을 거치면서 업데이트 된다.
       self.gpt = gpt
       if USE_ALPHA:
           self.bert = bert
           if USE_GAMMA:
               self.rogistic=torch.nn.Linear(bert_config.hidden_size,3)
           else:
               self.rogistic=torch.nn.Linear(bert_config.hidden_size,1)
           self.sigmoid=torch.nn.Sigmoid()
    #    self.grucell=nn.GRUCell(d_model,d_model).to(gpu_name)
       self.W1 = torch.nn.Linear(d_model, d_model, bias=False).to(gpu_name)
       self.W2 = torch.nn.Linear(d_model, d_model, bias=False).to(gpu_name)
       self.W3 = torch.nn.Linear(d_model, d_model, bias=False).to(gpu_name)
       self.W4 = torch.nn.Linear(d_model, d_model, bias=False).to(gpu_name)
       self.W5 = torch.nn.Linear(d_model, d_model, bias=False).to(gpu_name)

   def forward(self, memory,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask,labels,prev_predictions,conti_prev_predictions,conti_keyword_prev_predictions,order,whole,intro,tail,use_cumulative,use_memory,use_rake):#prompt_ids,prompt_attention):
       #memory states update.
       
        if debug :
            print("prev predictions : ")
            print(prev_predictions)
            print(prev_predictions.shape)
            print("conti_prev_predictions : ")
            print(conti_prev_predictions)
            print(conti_prev_predictions.shape)
            print("conti_keyword_prev_predictions : ")
            print(conti_keyword_prev_predictions)
            print(conti_keyword_prev_predictions.shape)
            print("input ids: ")
            print(input_ids)
            print(input_ids.shape)
            print("memory:")
            print(memory)
            print(memory.shape)
            print("attention_mask:")
            print(attention_mask)
            print(attention_mask.shape)
            print("decoder_input_ids:")
            print(decoder_input_ids)
            print(decoder_input_ids.shape)
            print("decoder_attention_mask:")
            print(decoder_attention_mask)
            print(decoder_attention_mask.shape)
            print("labels")
            print(labels)
            print(labels.shape)
            print("intro:")
            print(intro)
            print("tail:")
            print(tail)
            print("whole:")
            print(whole)
            print("order:")
            print(order)


        short_prev=prev_predictions # 뒤에서 사용
        
        
        
        if use_memory :
            prev_predictions = self.shared(prev_predictions)
            prev_predictions=torch.mean(prev_predictions,dim=1)
            prev_predictions=torch.unsqueeze(prev_predictions,dim=1)
            prev_predictions=prev_predictions.expand((-1,memory.shape[1],-1)) 
            print("prev_predictions shape:")
            print(prev_predictions.shape) # (b,400,d_model)
            
            #memory -> (b,400,d)
            #prev_prediction -> (b,400,d)
            #w mul and sum -> (b,400,d)
            Mhat = torch.tanh(self.W1(memory)+self.W2(prev_predictions))
            g = torch.sigmoid(self.W3(memory)+self.W4(prev_predictions))
            Mnext = torch.mul(1-g, memory) + torch.mul(g, Mhat)
            memory= F.normalize(1e-7+Mnext, dim = -1) # (b,400,d_model)
            # 주의!! 완전히 plotmachine 스타일이다.
            # 원래는 embedding avg가 아닌 그냥 prev_prediction 전체를 썼었다.
        else:
           memory=None
        print("after gru, memory : " )
        print(memory.shape)

        if use_cumulative :

            cumulation=self.shared(conti_prev_predictions) #(b,len,d)
            #print("cumulation's shape:")
            #print(cumulation.shape)
            cumulation=torch.tanh(self.W5(cumulation))
            #print("tanh cumulation's shape:")
            #print(cumulation.shape)
            cumulation=F.normalize(1e-7+cumulation, dim = -1)
            #cumulation=torch.mean(cumulation,dim=1) # (b,d)
            #cumulation=torch.unsqueeze(cumulation,dim=1) # (b,1,d)
            # 내친 김에 얘도 avg를 때려봤다.
        else:
            cumulation=None
        
        print("final cumulation's shape :")
        print(cumulation.shape) #(b,len,d_model)
        
        
        if intro:
            batch_decoding_token_tensors=batch_intro_token_tensors
        elif tail:
            batch_decoding_token_tensors=batch_tail_token_tensors
        else:
            batch_decoding_token_tensors=batch_body_token_tensors
        
    
        
        if order/whole<0.33 :
            batch_order_token_tensors=batch_front_token_tensors
        elif order/whole <0.66 :
            batch_order_token_tensors=batch_middle_token_tensors
        else:
            batch_order_token_tensors=batch_ending_token_tensors
        

        alpha=0.5
        beta=0.5
        if USE_ALPHA:
            if short_prev.shape[1]>500:
                short_prev=short_prev[:,-500:]
            print("previous shape : ")
            
            previous=torch.cat((short_prev,batch_eoprev_token_tensors,batch_decoding_token_tensors,batch_order_token_tensors,batch_next_is_ending_token_tensors),1)
            print(previous.shape)
            previous=tokenizer.batch_decode(previous,skip_special_tokens=True)
            print(previous)
            previous=bert_tokenizer(previous,max_length=300,padding="max_length",
            truncation=True,return_tensors="pt").input_ids.to(gpu_name)
            print(previous.shape)
            output=self.bert(previous)
            
            if USE_GAMMA:
                ratio=self.rogistic(output.pooler_output)
                ratio=self.sigmoid(ratio)
                alpha=torch.unsqueeze(torch.unsqueeze(ratio[:,0]/torch.sum(ratio,dim=1),dim=1),dim=1) #(b,1,1)
                beta=torch.unsqueeze(torch.unsqueeze(ratio[:,1]/torch.sum(ratio,dim=1),dim=1),dim=1)
                gamma=torch.unsqueeze(torch.unsqueeze(ratio[:,2]/torch.sum(ratio,dim=1),dim=1),dim=1)
            else:
                alpha=self.rogistic(output.pooler_output)
                alpha=self.sigmoid(alpha) 
                alpha=torch.unsqueeze(torch.mul((alpha),1/2),dim=1) #(b,1)
                #attention의 결과가 3 dim이면 (b,1,1)이어야함
                # 2 dim -> (b,1) 이 맞음
                beta=0.5-alpha

        if debug:
            print("alpha :")
            print(alpha)
            print("beta : ")
            print(beta)
            if USE_GAMMA:
                print("gamma : ")
                print(gamma)
        
        
        if use_rake is False or input_ids.shape[1]+conti_keyword_prev_predictions.shape[1]+5 > 1020 or intro:
            
        #    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,order_token_tensor,by_token_tensor,whole_token_tensor),1)
            if NO_IBT is False and NO_FME is False:
                if order==whole-1:
                    input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_decoding_token_tensors,batch_order_token_tensors,batch_next_is_ending_token_tensors),1) # 다음 문단이 ending이라는 정보를 알려준다.
                else:
                    input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_decoding_token_tensors,batch_order_token_tensors),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors),1)
            elif NO_FME:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_decoding_token_tensors),1)
            elif NO_IBT:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_order_token_tensors),1)
        else:
            if NO_IBT is False and NO_FME is False:
                if order==whole-1:
                    input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_soprev_token_tensors,conti_keyword_prev_predictions,batch_eoprev_token_tensors,batch_decoding_token_tensors,batch_order_token_tensors,batch_next_is_ending_token_tensors),1)
                else:
                    input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_soprev_token_tensors,conti_keyword_prev_predictions,batch_eoprev_token_tensors,batch_decoding_token_tensors,batch_order_token_tensors),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_soprev_token_tensors,conti_keyword_prev_predictions,batch_eoprev_token_tensors),1)
            elif NO_FME:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_soprev_token_tensors,conti_keyword_prev_predictions,batch_eoprev_token_tensors,batch_decoding_token_tensors),1)
            elif NO_IBT:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_soprev_token_tensors,conti_keyword_prev_predictions,batch_eoprev_token_tensors,batch_order_token_tensors),1)
        
        
        valid_input_ids=[]
        valid_labels=[]
        for b in range(batch_size):
            # GPT STYLE!! BART에서는 EOS 토큰을 뺄 필요도 없고, PAD만 날리면 되며, 애초에 DECODER INPUT IDS랑 합치지도 않는다.
            
            # 사실 bart에서는 pad를 날릴 필요조차 없다(어차피 attention mask) 즉 이 코드 전체가 gpt에서만 쓰인다.

            valid_position=torch.where((input_ids[b]!=tokenizer.pad_token_id) & (input_ids[b]!=tokenizer.eos_token_id))
            input_id=input_ids[b][valid_position]
            
            residual=input_id
            # gpt 토크나이저는 eos 토큰을 따로 추가하지 않는다. decoder input에만큼은 eos가 있어야 할 것 같다. eos 토큰을 추가해준다.
            input_id=torch.cat((input_id,decoder_input_ids[b],torch.LongTensor([tokenizer.eos_token_id]).to(gpu_name)),dim=0)
            padding=torch.LongTensor([tokenizer.pad_token_id]*(input_ids.shape[1]+conti_keyword_prev_predictions.shape[1]+decoder_input_ids.shape[1]+15-len(input_id))).to(gpu_name)
            print(input_id.shape)
            print(padding.shape)
            valid_input_ids.append(torch.cat((input_id,padding,),0))

            label=torch.cat((residual[1:],labels[b],torch.LongTensor([tokenizer.eos_token_id]).to(gpu_name)),dim=0)
            padding=torch.LongTensor([tokenizer.pad_token_id]*(input_ids.shape[1]+conti_keyword_prev_predictions.shape[1]+decoder_input_ids.shape[1]+15-len(label))).to(gpu_name)
            valid_labels.append(torch.cat((label,padding,),0))
        
        input_ids=torch.stack(valid_input_ids,dim=0)
        labels=torch.stack(valid_labels,dim=0)
        
        print("valid_input_ids.shape label과 차원이 같고 한칸씩 label이 밀려있어야 함.")
        print(input_ids)
        print(input_ids.shape)
        print("valid_labels.shape")
        print(labels)
        print(labels.shape)

        
        if debug:
            print("after preprocessing, input ids: ")
            print(tokenizer.batch_decode(input_ids,skip_special_tokens=False))
            print("after preprocessing, labels: ")
            print(tokenizer.batch_decode(labels,skip_special_tokens=False))

        inputs_embeds=self.shared(input_ids)
        print("input embeds shape : ")
        print(inputs_embeds.shape)

        # list_attention_mask=[[0]*input_ids.shape[1]]*batch_size

        # for i in range(input_ids.shape[1]):
        #     if input_ids[0][i]!=pad_token_id: # pad token id는 1이다. pad가 아니면 1로 해야 한다.
        #         list_attention_mask[0][i]=1
        attention_mask=torch.where(input_ids==tokenizer.pad_token_id,0,1).to(gpu_name)
        if debug:
            print("attention_mask")
            print(attention_mask)

        outputs = self.gpt(input_ids = None,inputs_embeds=inputs_embeds,attention_mask = attention_mask,labels=labels,output_hidden_states=True,memory=memory,context=cumulation,alpha=alpha,beta=beta)

        return outputs,memory
    
   def generate(self, memory,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask,labels,prev_predictions,conti_prev_predictions,conti_keyword_prev_predictions,order,whole,intro,tail,use_memory,use_cumulative,use_rake):#prompt_ids,prompt_attention):

       
        if debug :
            print("prev predictions : ")
            print(prev_predictions)
            print(prev_predictions.shape)
            print("conti_prev_predictions : ")
            print(conti_prev_predictions)
            print(conti_prev_predictions.shape)
            print("conti_keyword_prev_predictions : ")
            print(conti_keyword_prev_predictions)
            print(conti_keyword_prev_predictions.shape)
            print("input ids: ")
            print(input_ids)
            print(input_ids.shape)
            print("memory:")
            print(memory)
            print(memory.shape)
            print("attention_mask:")
            print(attention_mask)
            print(attention_mask.shape)
            print("decoder_input_ids:")
            print(decoder_input_ids)
            print(decoder_input_ids.shape)
            print("decoder_attention_mask:")
            print(decoder_attention_mask)
            print(decoder_attention_mask.shape)
            print("labels")
            print(labels)
            print(labels.shape)
            print("intro:")
            print(intro)
            print("tail:")
            print(tail)
            print("whole:")
            print(whole)
            print("order:")
            print(order)


        short_prev=prev_predictions # 뒤에서 사용
        
        
        
        if use_memory :
            prev_predictions = self.shared(prev_predictions)
            prev_predictions=torch.mean(prev_predictions,dim=1)
            prev_predictions=torch.unsqueeze(prev_predictions,dim=1)
            prev_predictions=prev_predictions.expand((-1,memory.shape[1],-1)) 
            print("prev_predictions shape:")
            print(prev_predictions.shape) # (b,400,d_model)
            
            #memory -> (b,400,d)
            #prev_prediction -> (b,400,d)
            #w mul and sum -> (b,400,d)
            Mhat = torch.tanh(self.W1(memory)+self.W2(prev_predictions))
            g = torch.sigmoid(self.W3(memory)+self.W4(prev_predictions))
            Mnext = torch.mul(1-g, memory) + torch.mul(g, Mhat)
            memory= F.normalize(1e-7+Mnext, dim = -1) # (b,400,d_model)
            # 주의!! 완전히 plotmachine 스타일이다.
            # 원래는 embedding avg가 아닌 그냥 prev_prediction 전체를 썼었다.
        else:
           memory=None
        print("after gru, memory : " )
        print(memory.shape)

        if use_cumulative :

            cumulation=self.shared(conti_prev_predictions) #(b,len,d)
            #print("cumulation's shape:")
            #print(cumulation.shape)
            cumulation=torch.tanh(self.W5(cumulation))
            #print("tanh cumulation's shape:")
            #print(cumulation.shape)
            cumulation=F.normalize(1e-7+cumulation, dim = -1)
            #cumulation=torch.mean(cumulation,dim=1) # (b,d)
            #cumulation=torch.unsqueeze(cumulation,dim=1) # (b,1,d)
            # 내친 김에 얘도 avg를 때려봤다.
        else:
            cumulation=None
        
        print("final cumulation's shape :")
        print(cumulation.shape) #(b,len,d_model)
        
        
        if intro:
            batch_decoding_token_tensors=batch_intro_token_tensors
        elif tail:
            batch_decoding_token_tensors=batch_tail_token_tensors
        else:
            batch_decoding_token_tensors=batch_body_token_tensors
        
    
        
        if order/whole<0.33 :
            batch_order_token_tensors=batch_front_token_tensors
        elif order/whole <0.66 :
            batch_order_token_tensors=batch_middle_token_tensors
        else:
            batch_order_token_tensors=batch_ending_token_tensors
        

        alpha=0.5
        beta=0.5
        if USE_ALPHA:
            if short_prev.shape[1]>500:
                short_prev=short_prev[:,-500:]
            print("previous shape : ")
            
            previous=torch.cat((short_prev,batch_eoprev_token_tensors,batch_decoding_token_tensors,batch_order_token_tensors,batch_next_is_ending_token_tensors),1)
            print(previous.shape)
            previous=tokenizer.batch_decode(previous,skip_special_tokens=True)
            print(previous)
            previous=bert_tokenizer(previous,max_length=300,padding="max_length",
            truncation=True,return_tensors="pt").input_ids.to(gpu_name)
            print(previous.shape)
            output=self.bert(previous)
            
            if USE_GAMMA:
                ratio=self.rogistic(output.pooler_output)
                ratio=self.sigmoid(ratio)
                alpha=torch.unsqueeze(torch.unsqueeze(ratio[:,0]/torch.sum(ratio,dim=1),dim=1),dim=1) #(b,1,1)
                beta=torch.unsqueeze(torch.unsqueeze(ratio[:,1]/torch.sum(ratio,dim=1),dim=1),dim=1)
                gamma=torch.unsqueeze(torch.unsqueeze(ratio[:,2]/torch.sum(ratio,dim=1),dim=1),dim=1)
            else:
                alpha=self.rogistic(output.pooler_output)
                alpha=self.sigmoid(alpha) 
                alpha=torch.unsqueeze(torch.mul((alpha),1/2),dim=1) #(b,1)
                #attention의 결과가 3 dim이면 (b,1,1)이어야함
                # 2 dim -> (b,1) 이 맞음
                beta=0.5-alpha

        if debug:
            print("alpha :")
            print(alpha)
            print("beta : ")
            print(beta)
            if USE_GAMMA:
                print("gamma : ")
                print(gamma)
       
         
       
        if use_rake is False or input_ids.shape[1]+conti_keyword_prev_predictions.shape[1]+5 > 1020 or intro:
            
        #    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,order_token_tensor,by_token_tensor,whole_token_tensor),1)
            if NO_IBT is False and NO_FME is False:
                if order==whole-1:
                    input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_decoding_token_tensors,batch_order_token_tensors,batch_next_is_ending_token_tensors),1) # 다음 문단이 ending이라는 정보를 알려준다.
                else:
                    input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_decoding_token_tensors,batch_order_token_tensors),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors),1)
            elif NO_FME:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_decoding_token_tensors),1)
            elif NO_IBT:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_order_token_tensors),1)
        else:
            if NO_IBT is False and NO_FME is False:
                if order==whole-1:
                    input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_soprev_token_tensors,conti_keyword_prev_predictions,batch_eoprev_token_tensors,batch_decoding_token_tensors,batch_order_token_tensors,batch_next_is_ending_token_tensors),1)
                else:
                    input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_soprev_token_tensors,conti_keyword_prev_predictions,batch_eoprev_token_tensors,batch_decoding_token_tensors,batch_order_token_tensors),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_soprev_token_tensors,conti_keyword_prev_predictions,batch_eoprev_token_tensors),1)
            elif NO_FME:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_soprev_token_tensors,conti_keyword_prev_predictions,batch_eoprev_token_tensors,batch_decoding_token_tensors),1)
            elif NO_IBT:
                input_ids=torch.cat((batch_soplot_token_tensors,input_ids,batch_eoplot_token_tensors,batch_soprev_token_tensors,conti_keyword_prev_predictions,batch_eoprev_token_tensors,batch_order_token_tensors),1)
                
        
        

        

            
        outputs=[]
        for b in range(batch_size):
            
        
            # GPT STYLE!! BART에서는 EOS 토큰을 뺄 필요도 없고, PAD만 날리면 되며, 애초에 DECODER INPUT IDS랑 합치지도 않는다.
            # 주의! gpt는 아예 eos 토큰이 있지도 않다.
            valid_position=torch.where((input_ids[b]!=tokenizer.pad_token_id) & (input_ids[b]!=tokenizer.eos_token_id))
            input_id=torch.unsqueeze(input_ids[b][valid_position],dim=0) #(1,dynamic_len)
            

            if debug:
                print("after preprocessing, input id: ")
                print(tokenizer.batch_decode(input_id,skip_special_tokens=False))
            # valid_input_ids.append(torch.cat((input_id,padding,),1))

            inputs_embeds=self.shared(input_id)
            print("input embeds shape : ")
            print(inputs_embeds.shape)

            attention_mask=torch.where(input_id==tokenizer.pad_token_id,0,1).to(gpu_name)
            if debug:
                print("attention_mask")
                print(attention_mask)
                print(attention_mask.shape)
            
            outputs.append(self.gpt.generate(max_length=250,memory=memory[b],inputs_embeds=inputs_embeds[b],attention_mask=attention_mask[b],
                    #num_beams=4,
                    do_sample=True,
                    top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
                    top_p=0.95,
                    no_repeat_ngram_size=3,
                    #encoder_no_repeat_ngram_size=3,
                    repetition_penalty=3.5,early_stopping=True,context=cumulation[b],alpha=alpha[b],beta=beta[b]))
        
        # outputs=torch.cat(outputs,dim=0)
        return outputs,memory

config = AutoConfig.from_pretrained('gpt2-medium')
bert_config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")

vocab_size=config.vocab_size
d_model=config.n_embd

if CONTINUOUSLY_TRAIN:
    gpt =  AutoModelForCausalLM.from_config(config).to(gpu_name) # 이후부터는 내가 finetune한 bart를 사용(밑에서 torch로 불러온다.)
    bert = AutoModel.from_config(bert_config).to(gpu_name)
else:
    gpt = AutoModelForCausalLM.from_pretrained('gpt2-medium').to(gpu_name) # 최초 학습에서는 pretrained 된 bart를 사용
    bert = AutoModel.from_pretrained("prajjwal1/bert-tiny").to(gpu_name)

gpt.resize_token_embeddings(len(tokenizer)) # 이렇게 하면 랜덤한 embedding unit이 추가가 된다.
bert.resize_token_embeddings(len(bert_tokenizer))
shared = gpt.get_input_embeddings()
shared.requires_grad = False
#bart.get_input_embeddings().requires_grad = False # embedding layer는 학습을 안한다. 얘가 변동되면 prev_predictions에 대한 표현도 계속 변하기 때문.
#생각해보니, shared에다가 init에서 복사한 embedding module만 계속 쓰는 거잖아?
model = Network(shared,vocab_size, d_model,gpt, bert,bert_config).to(gpu_name)

# -----------train ends, eval starts.
# f = open('rake_fme_second_level_val_results.csv','w', newline='')
# wr = csv.writer(f)
# wr.writerow(["steps","index","source","real text","generated_results"])

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-6)

if CONTINUOUSLY_TRAIN:
    checkpoint= torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# with open("pickle_data/"+"gpt_train_"+dataset_dir+"/level_2_whole.pickle","rb") as fi:
#             train_dataset = pickle.load(fi)

# num_training_steps = (num_epochs-1) * len(train_dataset) + len(train_dataset)-LAST_STEP


def trainer(LAST_STEP,train_dataset,NumPar):
    whole_count_for_save=0
    #do_eval(0)
    model.train()
    running_loss = 0.0
    
    use_cumulative=USE_CUMULATIVE
    use_memory=USE_MEMORY
    
    # print(batch_sep_token_tensors.shape)
    for i in range(LAST_STEP, len(train_dataset),batch_size):
    # get the inputs; data is a list of [inputs, labels]
        mini_running_loss=0.0
        batch_data=train_dataset[i:i+batch_size]
        first=True
        batch_num_decoder_input_ids=[]
        batch_decoder_attention_masks=[]
        for data in batch_data:
            input_ids,attention_mask,num_decoder_input_ids,decoder_attention_masks,prompt= (data['input_ids'],data['input_attention'],data['decoder_input_ids'],data['decoder_attention_mask'],data['prompt'])
            batch_num_decoder_input_ids.append(num_decoder_input_ids)# 또 각각 decoder_input_ids에서도 <s>는 떼내야 한다.
            #이건 데이터셋 만들때 처리해줘야 할듯하다.
            batch_decoder_attention_masks.append(decoder_attention_masks)
            if first:
                batch_input_ids=input_ids # 생각해보니 input_ids에서 </s>를 떼야 될 것 같다.
                batch_attention_mask=attention_mask
                batch_prev_predictions=prompt
                first=False
            else:
                batch_input_ids=torch.cat((batch_input_ids,input_ids),dim=0)
                batch_attention_mask=torch.cat((batch_attention_mask,attention_mask),dim=0)
                
                batch_prev_predictions=torch.cat((batch_prev_predictions,prompt),dim=0)
        batch_num_decoder_input_ids=torch.stack(batch_num_decoder_input_ids,dim=1)
        batch_decoder_attention_masks=torch.stack(batch_decoder_attention_masks,dim=1)
        print("batch dataset shapes")
        print(batch_input_ids.shape) #(b, 200)
        print(batch_attention_mask.shape) 
        print(batch_num_decoder_input_ids.shape) # (N,b,250)
        print(batch_decoder_attention_masks.shape) # (N, b, 250)
        print(batch_prev_predictions.shape) #(b,150)
        print("batch dataset shape ends")
        batch_input_ids=batch_input_ids.to(gpu_name)
        batch_attention_mask=batch_attention_mask.to(gpu_name)
        batch_num_decoder_input_ids=batch_num_decoder_input_ids.to(gpu_name)
        batch_decoder_attention_masks=batch_decoder_attention_masks.to(gpu_name)
        batch_prev_predictions=batch_prev_predictions.to(gpu_name)
        
        count=0
    
        
        emb_input_ids = shared(batch_input_ids)
        print("emb input ids shape : ")
        print(emb_input_ids.shape)

        memory = torch.zeros_like(torch.empty(batch_size,batch_input_ids.shape[1],d_model)).to(gpu_name) # first memory.
        memory = torch.cat((emb_input_ids,memory),dim=1) # (B,400,1024)
        #cumul_prev_predictions = torch.zeros_like(torch.empty(1,1)).to(gpu_name)
        batch_cumul_prev_predictions=[]
        batch_keyword_prev_predictions=[]
        batch_conti_keyword_prev_predictions=torch.zeros_like(torch.empty(batch_size,1),dtype=torch.long).to(gpu_name)
        batch_conti_prev_predictions=torch.zeros_like(torch.empty(batch_size,1),dtype=torch.long).to(gpu_name)
        
    #print(prev_predictions)
        #torch.cuda.empty_cache() # manually freeing gpu memory.
        for count in range(NumPar):

            _batch_decoder_input_ids=batch_num_decoder_input_ids[count] #(b,250)
            _batch_decoder_attention_masks=batch_decoder_attention_masks[count] #(b,250)

            
            # dd=torch.unsqueeze(decoder_input_id[:-1],dim=0).to(gpu_name)
            # decoder_attention_mask=torch.unsqueeze(decoder_attention_masks[count][:-1],dim=0).to(gpu_name)
        # input_ids 맨 앞에 이전 preceding context를 합친다.
            # label=torch.unsqueeze(d[1:],dim=0).to(gpu_name)

            # _batch_labels=_batch_decoder_input_ids[:,1:] #(b,249)
            _batch_labels=_batch_decoder_input_ids #(b,250)
            # 주의!!! gpt는 이렇게 하지만 bart는 위에 코드로 해야함!
            _batch_decoder_input_ids=_batch_decoder_input_ids[:,:-1] #(b,249)
            _batch_decoder_attention_masks=_batch_decoder_attention_masks[:,:-1] #(b,249)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            if len(batch_cumul_prev_predictions)>0:
                batch_conti_prev_predictions=batch_cumul_prev_predictions[0] #(b,~)
                batch_conti_keyword_prev_predictions=batch_keyword_prev_predictions[0] #(b,~)

            if use_cumulative and count>0:
                length=len(batch_cumul_prev_predictions)
                #print("one step." + str(length))
                for j in range(1,CUMUL_NUM if length>CUMUL_NUM else length):
                    #print(prev_predictions.shape)
                    if batch_input_ids.shape[1]+(batch_cumul_prev_predictions[j].shape[1])+batch_conti_prev_predictions.shape[1]>1000:
                        #print("break")
                        #print(cumul_prev_predictions[j].shape)
                        break
                    batch_conti_prev_predictions=torch.cat((batch_conti_prev_predictions,batch_sep_token_tensors,batch_cumul_prev_predictions[j]),1)       
                    batch_conti_keyword_prev_predictions=torch.cat((batch_conti_keyword_prev_predictions,batch_sep_token_tensors,batch_keyword_prev_predictions[j]),1)
            
            intro=False
            tail=False
            
            if count==0:
                intro=True
                if USE_FUSION is True:
                    use_memory=True
                    ## fusion ver.
            
            if USE_FUSION is True:
                use_cumulative=False ## fusion ver.
            
            if count==len(num_decoder_input_ids)-1:
                tail=True
                intro=False
                if USE_FUSION is True:
                    use_memory=False
                    use_cumulative=True
                    ## fusion ver.


            order=count+1 #order는 1부터 시작한다.
            whole=NumPar
            batch_conti_prev_predictions=batch_conti_prev_predictions.to(gpu_name) #(b,~)
            batch_conti_keyword_prev_predictions=batch_conti_keyword_prev_predictions.to(gpu_name) #(b,~)
            print("batch conti prev predction과 batch conti keyword prev prediction shape.")
            print(batch_conti_prev_predictions.shape)
            print(batch_conti_keyword_prev_predictions.shape)

            outputs,new_memory = model(memory=memory.detach(),input_ids = batch_input_ids,attention_mask = batch_attention_mask,decoder_input_ids = _batch_decoder_input_ids,decoder_attention_mask=_batch_decoder_attention_masks,labels=_batch_labels,prev_predictions=batch_prev_predictions,
                                conti_prev_predictions=batch_conti_prev_predictions,conti_keyword_prev_predictions=batch_conti_keyword_prev_predictions,order=order,whole=whole,intro=intro,tail=tail,use_cumulative=use_cumulative,use_memory=use_memory,use_rake=USE_RAKE)#prompt_ids=prompt_ids,prompt_attention=prompt_attention) # 중요! memory.detach()를 하지 않으면 매번 memory cell에 대한 gradient는 계속 이어져나가 계산되기 때문에, 두번 그래디언트 업데이트 했다고 오류 뜬다.
            
            if use_memory is True:
                memory=new_memory
            

            loss = outputs.loss
            loss.backward()

            
            batch_prev_predictions =  torch.argmax(outputs.logits, dim=-1) #(b,output_seq_len)
            
            # 이게 잘못됐다. teacher forcing일땐 상관 없는데.
            # gpt에서는 output이 input을 싸그리 포함하고 있으므로,
            # output에서 input 부분만을 빼고 보는 게 필요하다
            batch_prev_predictions=batch_prev_predictions[:,-249:]

            print("batch output(물론 teacher forcing하면 250이 될거임)") 
            print(batch_prev_predictions)           
            print(batch_prev_predictions.shape) #(아마도 b,1024)?
            if TEACHER_FORCING_MEMORY:
                batch_prev_predictions = _batch_labels # teacher forcing으로, memory와 cumul에 쓰이는 prev prediction은 training 과정에선 golden label 사용!
                #(b,250)
                print("teacher forcing 하면 batch prev predition:")
                print(batch_prev_predictions.shape)
            if USE_FUSION is True:
                use_cumulative=True ## fusion ver.

            if use_cumulative:
                batch_cumul_prev_predictions.insert(0,batch_prev_predictions) # (b,250)
            
            
            if use_cumulative:
                texts_prev_predictions=tokenizer.batch_decode(batch_prev_predictions,skip_special_tokens=True)
                
                batch_keywordsSTR=[]
                for text in texts_prev_predictions:
                    # print("texts for rake")
                    # print(text)
                    r.extract_keywords_from_text(text)
                    top_features= r.get_ranked_phrases()
                    topK=5

                    if len(top_features)==0:
                        keywordsSTR="[SEP]"
                    else:
                        top_features = clean_top_features(top_features, topK)
                        keywordsSTR = convert_keys_to_str(top_features)
                    # print("keywords STR")
                    # print(keywordsSTR)
                    batch_keywordsSTR.append(keywordsSTR)

                batch_keyword_prev_predictions.insert(0,tokenizer(batch_keywordsSTR,max_length=50,padding="max_length",
            truncation=True,return_tensors='pt').input_ids.to(gpu_name))
                
                

                if debug:
                    print("batch_keyword_prev_predictions shape")
                    print(len(batch_keyword_prev_predictions))
                    print("keywords from last output:")
                    print(batch_keywordsSTR)
            if debug:    
                input()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    # print statistics
            mini_running_loss += loss.item()
        
        running_loss +=mini_running_loss / count+1
        progress_bar.update(1)
        whole_count_for_save+=1
        if i % 3000 == 2999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / whole_count_for_save :.8f}, torch saved.')
            
            writer.add_scalar("Loss/train", running_loss/whole_count_for_save, epoch * len(train_dataset) + i)
            
            running_loss=0.0
            whole_count_for_save=0
            torch.save({'epoch':num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },PATH)

        
    # do_eval(epoch * len(train_dataset)+i)
    writer.flush()
    print('Finished - ' + str(NumPar) + ' - Training')

    torch.save({'epoch':num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },PATH)


def do_eval(steps,dataset,NumPar):
    f = open(save_dir+'_generations_outputs.csv','w',encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(["steps","index","source","real text","generated_results"])
    index=0
    N=100
    # 이건 문단 내부적으로  얼마나 반복성이 심한지 보는 지표이다.
    in_self_bleu_one=0
    in_self_bleu_bi=0
    in_self_bleu_tri=0
    in_self_bleu_four=0
    in_self_bleu_fif=0
    

    self_bleu_one=0
    self_bleu_bi=0
    self_bleu_tri=0
    self_bleu_four=0
    self_bleu_fif=0
    r_self_bleu_one=0
    r_self_bleu_bi=0
    r_self_bleu_tri=0
    r_self_bleu_four=0
    r_self_bleu_fif=0
    whole_num=0
    whole_predictions=[]
    whole_predictions_len=0
    whole_labels=[]
    whole_labels_len=0

    whole_nlls = []

    use_cumulative=USE_CUMULATIVE
    use_memory=USE_MEMORY

    model.eval()
    for i in trange(0, 100, batch_size):
    # get the inputs; data is a list of [inputs, labels]]
        batch_data=dataset[i:i+batch_size]
        first=True
        batch_num_decoder_input_ids=[]
        batch_decoder_attention_masks=[]
        one_prediction=[]
        for i in range(batch_size):
            one_prediction.append([])
        one_label=[]
        for j in range(batch_size):
            one_label.append([])
        for data in batch_data:
            input_ids,attention_mask,num_decoder_input_ids,decoder_attention_masks,prompt= (data['input_ids'],data['input_attention'],data['decoder_input_ids'],data['decoder_attention_mask'],data['prompt'])
            batch_num_decoder_input_ids.append(num_decoder_input_ids)
            batch_decoder_attention_masks.append(decoder_attention_masks)
            if first:
                batch_input_ids=input_ids
                batch_attention_mask=attention_mask
                batch_prev_predictions=prompt
                first=False
            else:
                batch_input_ids=torch.cat((batch_input_ids,input_ids),dim=0)
                batch_attention_mask=torch.cat((batch_attention_mask,attention_mask),dim=0)
                
                batch_prev_predictions=torch.cat((batch_prev_predictions,prompt),dim=0)
        batch_num_decoder_input_ids=torch.stack(batch_num_decoder_input_ids,dim=1)
        batch_decoder_attention_masks=torch.stack(batch_decoder_attention_masks,dim=1)
        print("batch dataset shapes")
        print(batch_input_ids.shape) #(b, 200)
        print(batch_attention_mask.shape) 
        print(batch_num_decoder_input_ids.shape) # (N,b,250)
        print(batch_decoder_attention_masks.shape) # (N, b, 250)
        print(batch_prev_predictions.shape) #(b,150)
        print("batch dataset shape ends")
        batch_input_ids=batch_input_ids.to(gpu_name)
        batch_attention_mask=batch_attention_mask.to(gpu_name)
        batch_num_decoder_input_ids=batch_num_decoder_input_ids.to(gpu_name)
        batch_decoder_attention_masks=batch_decoder_attention_masks.to(gpu_name)
        batch_prev_predictions=batch_prev_predictions.to(gpu_name)
        
        count=0
    
        
        emb_input_ids = shared(batch_input_ids)
        print("emb input ids shape : ")
        print(emb_input_ids.shape)

        memory = torch.zeros_like(torch.empty(batch_size,batch_input_ids.shape[1],d_model)).to(gpu_name) # first memory.
        memory = torch.cat((emb_input_ids,memory),dim=1) # (B,400,1024)
        #cumul_prev_predictions = torch.zeros_like(torch.empty(1,1)).to(gpu_name)
        batch_cumul_prev_predictions=[]
        batch_keyword_prev_predictions=[]
        batch_conti_keyword_prev_predictions=torch.zeros_like(torch.empty(batch_size,1),dtype=torch.long).to(gpu_name)
        batch_conti_prev_predictions=torch.zeros_like(torch.empty(batch_size,1),dtype=torch.long).to(gpu_name)
        
    #print(prev_predictions)
        #torch.cuda.empty_cache() # manually freeing gpu memory.
        for count in range(NumPar):

            _batch_decoder_input_ids=batch_num_decoder_input_ids[count] #(b,250)
            _batch_decoder_attention_masks=batch_decoder_attention_masks[count] #(b,250)

            
            # dd=torch.unsqueeze(decoder_input_id[:-1],dim=0).to(gpu_name)
            # decoder_attention_mask=torch.unsqueeze(decoder_attention_masks[count][:-1],dim=0).to(gpu_name)
        # input_ids 맨 앞에 이전 preceding context를 합친다.
            # label=torch.unsqueeze(d[1:],dim=0).to(gpu_name)
            # _batch_labels=_batch_decoder_input_ids[:,1:] #(b,249)
            _batch_labels=_batch_decoder_input_ids #(b,250)
            _batch_decoder_input_ids=_batch_decoder_input_ids[:,:-1] #(b,249)
            _batch_decoder_attention_masks=_batch_decoder_attention_masks[:,:-1] #(b,249)

            
            if len(batch_cumul_prev_predictions)>0:
                batch_conti_prev_predictions=batch_cumul_prev_predictions[0] #(b,~)
                batch_conti_keyword_prev_predictions=batch_keyword_prev_predictions[0] #(b,~)

            if use_cumulative and count>0:
                length=len(batch_cumul_prev_predictions)
                #print("one step." + str(length))
                for j in range(1,CUMUL_NUM if length>CUMUL_NUM else length):
                    #print(prev_predictions.shape)
                    if batch_input_ids.shape[1]+(batch_cumul_prev_predictions[j].shape[1])+batch_conti_prev_predictions.shape[1]>1000:
                        #print("break")
                        #print(cumul_prev_predictions[j].shape)
                        break
                    batch_conti_prev_predictions=torch.cat((batch_conti_prev_predictions,batch_sep_token_tensors,batch_cumul_prev_predictions[j]),1)       
                    batch_conti_keyword_prev_predictions=torch.cat((batch_conti_keyword_prev_predictions,batch_sep_token_tensors,batch_keyword_prev_predictions[j]),1)
            
            intro=False
            tail=False
            
            if count==0:
                intro=True
                if USE_FUSION is True:
                    use_memory=True
                    ## fusion ver.
            
            if USE_FUSION is True:
                use_cumulative=False ## fusion ver.
            
            if count==len(num_decoder_input_ids)-1:
                tail=True
                intro=False
                if USE_FUSION is True:
                    use_memory=False
                    use_cumulative=True
                    ## fusion ver.


            order=count+1 #order는 1부터 시작한다.
            whole=NumPar
            batch_conti_prev_predictions=batch_conti_prev_predictions.to(gpu_name) #(b,~)
            batch_conti_keyword_prev_predictions=batch_conti_keyword_prev_predictions.to(gpu_name) #(b,~)
            print("batch conti prev predction과 batch conti keyword prev prediction shape.")
            print(batch_conti_prev_predictions.shape)
            print(batch_conti_keyword_prev_predictions.shape)
            

            outputs,new_memory = model.generate(memory=memory.detach(),input_ids = batch_input_ids,attention_mask = batch_attention_mask,decoder_input_ids = _batch_decoder_input_ids,decoder_attention_mask=_batch_decoder_attention_masks,labels=_batch_labels,prev_predictions=batch_prev_predictions,
                                conti_prev_predictions=batch_conti_prev_predictions,conti_keyword_prev_predictions=batch_conti_keyword_prev_predictions,order=order,whole=whole,intro=intro,tail=tail,use_cumulative=use_cumulative,use_memory=use_memory,use_rake=USE_RAKE)#prompt_ids=prompt_ids,prompt_attention=prompt_attention) # 중요! memory.detach()를 하지 않으면 매번 memory cell에 대한 gradient는 계속 이어져나가 계산되기 때문에, 두번 그래디언트 업데이트 했다고 오류 뜬다.
            
            if use_memory is True:
                memory=new_memory
            
            whole_output=[]
            for output in outputs:
                print("개별 generation output의 shape -> batch 1이 없을 지도?")
                print(output.shape)
                padding=torch.LongTensor([[tokenizer.pad_token_id]*(300-output.shape[1])]).to(gpu_name)
                whole_output.append(torch.cat((output,padding,),1))

            
            batch_prev_predictions=torch.cat(whole_output,dim=0)

            print("generation output shape : ")
            print(batch_prev_predictions.shape) # (b,300)
            predictions = tokenizer.batch_decode(batch_prev_predictions,skip_special_tokens=True)
            labels = tokenizer.batch_decode(_batch_labels,skip_special_tokens=True)
            print("batch output")            
            print(batch_prev_predictions.shape) #(아마도 b,1024)?
            
            
            if USE_FUSION is True:
                use_cumulative=True ## fusion ver.

            if use_cumulative:
                batch_cumul_prev_predictions.insert(0,batch_prev_predictions) # (b,250)
            
            
            if use_cumulative:
                texts_prev_predictions=tokenizer.batch_decode(batch_prev_predictions,skip_special_tokens=True)
                
                batch_keywordsSTR=[]
                for text in texts_prev_predictions:
                    r.extract_keywords_from_text(text)
                    top_features= r.get_ranked_phrases()
                    topK=5

                    if len(top_features)==0:
                        keywordsSTR="[SEP]"
                    else:
                        top_features = clean_top_features(top_features, topK)
                        keywordsSTR = convert_keys_to_str(top_features)
                    
                    batch_keywordsSTR.append(keywordsSTR)

                batch_keyword_prev_predictions.insert(0,tokenizer(batch_keywordsSTR,max_length=50,padding="max_length",
            truncation=True,return_tensors='pt').input_ids.to(gpu_name))
                
                

                if debug:
                    print("keywords from last output:")
                    print(batch_keywordsSTR)
            
            if debug:
                print("-----------")
                print("predictions")
                print(predictions[0]) 
                
                print("golden label")
                print(labels[0]) #batch 중 1개만 본다.
                input()


            for u,pred in enumerate(predictions):
                
                _pred= tokenizer(pred,return_tensors="pt")['input_ids']
                _predictions_len+=len(_pred[0])
                one_prediction[u].append(pred)
            
            for u,lab in enumerate(labels):
                _lab= tokenizer(lab,return_tensors="pt")['input_ids']
                _labels_len+=len(_lab[0])
                one_label[u].append(lab)

            batch_input_text=tokenizer.batch_decode(batch_input_ids,skip_special_tokens=True)
            for b in batch_size:
                wr.writerow([str(steps),str(index),batch_input_text[b],labels[b],predictions[b]]) # 전부 plain string이다.
                index+=1
        

        
        for u,_one_prediction in enumerate(one_prediction):
            _in_self_bleu_one=0
            _in_self_bleu_bi=0
            _in_self_bleu_tri=0
            _in_self_bleu_four=0
            _in_self_bleu_fif=0

            whole_one_prediction=' '.join(_one_prediction)
            whole_one_label=' '.join(one_label[u])

            if len(_one_prediction)>1:
                for j in range(len(_one_prediction)): 
                    except_one_prediction=_one_prediction[0:j]+_one_prediction[j+1:]
                
            #self_bleu=BLEU(except_whole_predictions,weights).get_score([whole_predictions[j]])
                    self_bleu=_bleu.compute(predictions=[_one_prediction[j]],references=[except_one_prediction],max_order=5)
                    _in_self_bleu_one+=self_bleu['precisions'][0]
                    _in_self_bleu_bi+=self_bleu['precisions'][1]
                    _in_self_bleu_tri+=self_bleu['precisions'][2]
                    _in_self_bleu_four+=self_bleu['precisions'][3]
                    _in_self_bleu_fif+=self_bleu['precisions'][4]

            in_self_bleu_one+=_in_self_bleu_one/len(_one_prediction)
            in_self_bleu_bi+=_in_self_bleu_bi/len(_one_prediction)
            in_self_bleu_tri+=_in_self_bleu_tri/len(_one_prediction)
            in_self_bleu_four+=_in_self_bleu_four/len(_one_prediction)
            in_self_bleu_fif+=_in_self_bleu_fif/len(_one_prediction)
        
        
            if len(whole_one_label)==0:
                print("something wrong. label is not exist.")
                print("error set : ")
                print("predictions : ")
                print(whole_one_prediction)
                print("label : ")
                print(whole_one_label)
                continue
            whole_labels.append(whole_one_label)
            whole_labels_len+=_labels_len
            whole_predictions.append(whole_one_prediction)
            whole_predictions_len+=_predictions_len
            metric.add_batch(predictions=[whole_one_prediction], references=[whole_one_label])
            whole_num+=1
        

    result=metric.compute()
    print(result)
   
    ppl=0
    #print("ppl is : " + str(ppl.item()))
    gpt.cpu()
    bert.cpu()
    model.cpu()
    #del model, bart, bert
    #gc.collect()
    #torch.cuda.empty_cache()
    # 이제 모델은 필요 없으니 free 해준다.

    in_self_bleu_one=in_self_bleu_one/whole_num
    in_self_bleu_bi=in_self_bleu_bi/whole_num
    in_self_bleu_tri=in_self_bleu_tri/whole_num
    in_self_bleu_four=in_self_bleu_four/whole_num
    in_self_bleu_fif=in_self_bleu_fif/whole_num
    print("in_self_bleu_one : " + str(in_self_bleu_one))
    print("in_self_bleu_bi : " + str(in_self_bleu_bi))
    print("in_self_bleu_tri : " + str(in_self_bleu_tri))
    print("in_self_bleu_four : " + str(in_self_bleu_four))
    print("in_self_bleu_fif : " + str(in_self_bleu_fif))


    print("len of sample generation : " + str(whole_num))
    
    for j in range(N): # 1000개에 대해서만 self-bleu.
        except_whole_predictions=whole_predictions[0:j]+whole_predictions[j+1:1000]
        
        #self_bleu=BLEU(except_whole_predictions,weights).get_score([whole_predictions[j]])
        self_bleu=_bleu.compute(predictions=[whole_predictions[j]],references=[except_whole_predictions],max_order=5)
        self_bleu_one+=self_bleu['precisions'][0]
        self_bleu_bi+=self_bleu['precisions'][1]
        self_bleu_tri+=self_bleu['precisions'][2]
        self_bleu_four+=self_bleu['precisions'][3]
        self_bleu_fif+=self_bleu['precisions'][4]
    
    self_num=0
    for j in range(N):
        except_whole_labels=whole_labels[0:j]+whole_labels[j+1:1000]
        real_self_bleu=_bleu.compute(predictions=[whole_labels[j]],references=[except_whole_labels],max_order=5)
        r_self_bleu_one+=real_self_bleu['precisions'][0]
        r_self_bleu_bi+=real_self_bleu['precisions'][1]
        r_self_bleu_tri+=real_self_bleu['precisions'][2]
        r_self_bleu_four+=real_self_bleu['precisions'][3]
        r_self_bleu_fif+=real_self_bleu['precisions'][4]
            
        #print(_real_self_bleu)
        self_num+=1

    
    whole_predictions_len=whole_predictions_len/whole_num
    whole_labels_len=(whole_labels_len/whole_num)
    self_bleu_one=self_bleu_one/N
    self_bleu_bi=self_bleu_bi/N
    self_bleu_tri=self_bleu_tri/N
    self_bleu_four=self_bleu_four/N
    self_bleu_fif=self_bleu_fif/N
    r_self_bleu_one=r_self_bleu_one/N
    r_self_bleu_bi=r_self_bleu_bi/N
    r_self_bleu_tri=r_self_bleu_tri/N
    r_self_bleu_four=r_self_bleu_four/N
    r_self_bleu_fif=r_self_bleu_fif/N

    print("avg prediction len : " + str(whole_predictions_len))
    print("self_bleu one : " + str(self_bleu_one))
    print("self_bleu bi : " + str(self_bleu_bi))
    print("self_bleu tri : " + str(self_bleu_tri))
    print("self_bleu four : " + str(self_bleu_four))
    print("self_bleu fif : " + str(self_bleu_fif))
    print("avg reference len : " + str(whole_labels_len))
    print("real self_bleu one : " + str(r_self_bleu_one))
    print("real self_bleu bi : " + str(r_self_bleu_bi))
    print("real self_bleu tri : " + str(r_self_bleu_tri))
    print("real self_bleu four : " + str(r_self_bleu_four))
    print("real self_bleu fif : " + str(r_self_bleu_fif))
    

    writer.add_scalar("rouge1/eval", result['rouge1'], steps)
    writer.add_scalar("rouge2/eval", result['rouge2'], steps)
    writer.add_scalar("rougeL/eval", result['rougeL'], steps)
    writer.add_scalar("rougeLsum/eval", result['rougeLsum'], steps)
    writer.add_scalar("in self bleu one/eval",in_self_bleu_one, steps)
    writer.add_scalar("in self bleu bi/eval", in_self_bleu_bi, steps)
    writer.add_scalar("in self bleu tri/eval", in_self_bleu_tri, steps)
    writer.add_scalar("in self bleu four/eval", in_self_bleu_four, steps)
    writer.add_scalar("in self bleu fif/eval",in_self_bleu_fif, steps)
    writer.add_scalar("self bleu bi/eval", self_bleu_bi, steps)
    writer.add_scalar("self bleu tri/eval", self_bleu_tri, steps)
    writer.add_scalar("self bleu four/eval", self_bleu_four, steps)
    writer.add_scalar("self bleu fif/eval", self_bleu_fif, steps)
    writer.add_scalar("real_self bleu bi/eval", r_self_bleu_bi, steps)
    writer.add_scalar("real_self bleu tri/eval", r_self_bleu_tri, steps)
    writer.add_scalar("real_self bleu four/eval", r_self_bleu_four, steps)
    writer.add_scalar("real_self bleu fif/eval", r_self_bleu_fif, steps)
    #writer.add_scalar("meteor",met_result,steps)
    writer.add_scalar("predictions avg len",whole_predictions_len,steps)
    writer.add_scalar("references avg len",whole_labels_len,steps)
    #writer.add_scalar("ppl",ppl.item(),steps)

for epoch in range(num_epochs):  # loop over the dataset multiple times

    for i in range(LAST_PARAG,30): # 최대 30개 문단까지 있다.

        with open("pickle_data/"+"gpt_train_"+dataset_dir+"/level_2_" + str(i) + ".pickle","rb") as fi:
                train_dataset = pickle.load(fi)
        num_training_steps = (num_epochs-1) * len(train_dataset) + len(train_dataset)-LAST_STEP
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=20000, num_training_steps=num_training_steps
        )
        
        progress_bar = tqdm(range(num_training_steps))

        
        
        trainer(LAST_STEP,train_dataset=train_dataset,NumPar=i)
        writer.close()
        LAST_STEP=0
    
    for i in range(LAST_PARAG,30): # 최대 30개 문단까지 있다.

        valid_dataset_dir=""
        if dataset_dir=="wp_rake":
            valid_dataset_dir="gpt_valid_wp_rake"
        else:
            valid_dataset_dir="gpt_valid_"+dataset_dir
        with open("pickle_data/"+valid_dataset_dir+"/level_2_"+str(i)+".pickle","rb") as fi:
                valid_dataset = pickle.load(fi)
        
        do_eval(steps=epoch,dataset=valid_dataset,NumPar=i)
    



    


