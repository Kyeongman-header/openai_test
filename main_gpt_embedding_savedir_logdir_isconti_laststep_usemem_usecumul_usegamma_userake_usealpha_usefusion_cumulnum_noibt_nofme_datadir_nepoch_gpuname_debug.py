

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

save_dir=sys.argv[1] # rake_all
log_dir=sys.argv[2] # rake_all

conti=int(sys.argv[3]) # 0 

last_step=int(sys.argv[4]) # 0
use_mem=int(sys.argv[5]) # 1
use_cumul=int(sys.argv[6]) # 1
use_rake=int(sys.argv[7])
use_alpha=int(sys.argv[8])
use_fusion=int(sys.argv[9])

cumul_num=int(sys.argv[10]) # 3

no_ibt=int(sys.argv[11])
no_fme=int(sys.argv[12])

dataset_dir=sys.argv[13]
num_epochs=int(sys.argv[14])

gpu_name=sys.argv[15] # cuda:0
debug = int(sys.argv[16]) # 1
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


#with open("pickle_data/"+"train"+"/level_2.pickle","rb") as fi:
with open("pickle_data/"+"gpt_train_"+dataset_dir+"/level_2.pickle","rb") as fi:
        train_dataset = pickle.load(fi)
valid_dataset_dir=""
if dataset_dir=="wp_rake":
    valid_dataset_dir="gpt_val_wp_rake"
else:
    valid_dataset_dir="gpt_valid_"+dataset_dir
with open("pickle_data/"+valid_dataset_dir+"/level_2.pickle","rb") as fi:
        valid_dataset = pickle.load(fi)



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

b_soprev_token_tensor=torch.LongTensor([[soprev_id]]).to(gpu_name)
b_eoprev_token_tensor=torch.LongTensor([[eoprev_id]]).to(gpu_name)
b_sep_token_tensor=torch.LongTensor([[sep_id]]).to(gpu_name)
# by_token_tensor=torch.LongTensor([[by_id]]).to(gpu_name)
b_intro_token_tensor=torch.LongTensor([[intro_id]]).to(gpu_name)
b_body_token_tensor=torch.LongTensor([[body_id]]).to(gpu_name)
b_tail_token_tensor=torch.LongTensor([[tail_id]]).to(gpu_name)
b_front_token_tensor=torch.LongTensor([[front_id]]).to(gpu_name)
b_middle_token_tensor=torch.LongTensor([[middle_id]]).to(gpu_name)
b_ending_token_tensor=torch.LongTensor([[ending_id]]).to(gpu_name)
b_next_is_ending_token_tensor=torch.LongTensor([[next_is_ending_id]]).to(gpu_name)

class Network(nn.Module): 
   def __init__(self, vocab_size, d_model,gpt,bert,bert_config): 
       super(Network, self).__init__() 
        
       self.shared = gpt.get_input_embeddings()
       #nn.Embedding(config.vocab_size, config.d_model) 
       self.shared.requires_grad = False # 이 shared는 역할상 고정되어 있어야 한다.
       # 하지만 bart의 embedding layer는 학습을 거치면서 업데이트 된다.
       self.gpt = gpt
       self.bert = bert
       self.rogistic=torch.nn.Linear(bert_config.hidden_size,1)
       self.sigmoid=torch.nn.Sigmoid()
       self.grucell=nn.GRUCell(d_model,d_model).to(gpu_name)
       """
       self.wHr = nn.Linear(d_model,d_model).to(gpu_name)
       self.wMr = nn.Linear(d_model,d_model).to(gpu_name)
       self.wHz = nn.Linear(d_model,d_model).to(gpu_name)
       self.wMz = nn.Linear(d_model,d_model).to(gpu_name)
       self.wHn = nn.Linear(d_model,d_model).to(gpu_name)
       self.wMn = nn.Linear(d_model,d_model).to(gpu_name)
       """

   def forward(self, memory,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask,labels,prev_predictions,conti_prev_predictions,conti_keyword_prev_predictions,order,whole,intro,tail,use_cumulative,use_memory,use_rake):#prompt_ids,prompt_attention):
       #memory states update.
       
        if debug :
            print("prev predictions : ")
            print(prev_predictions)
            print("conti_prev_predictions : ")
            print(conti_prev_predictions)
            print("input ids: ")
            print(input_ids)
            print("memory:")
            print(memory)
            print("attention_mask:")
            print(attention_mask)
            print("decoder_input_ids:")
            print(decoder_input_ids)
            print("decoder_attention_mask:")
            print(decoder_attention_mask)
            print("labels")
            print(labels)
            print("intro:")
            print(intro)
            print("tail:")
            print(tail)
            print("whole:")
            print(whole)
            print("order:")
            print(order)


        short_prev=prev_predictions # 뒤에서 사용
        prev_predictions=torch.cat((torch.LongTensor([[tokenizer.pad_token_id]*(1024-prev_predictions.shape[1])]).to(gpu_name),prev_predictions),1)
        
        prev_predictions = self.shared(prev_predictions)
       #print(prev_predictions.shape)
        
        if use_memory :
           memory=self.grucell(torch.squeeze(prev_predictions),torch.squeeze(memory)).unsqueeze(dim=0)
        else:
           memory=None
        
        if use_cumulative :
            cumulation=self.shared(conti_prev_predictions)
        else:
            cumulation=None
        
        
        
        if intro:
            decoding_token_tensor=intro_token_tensor
        elif tail:
            decoding_token_tensor=tail_token_tensor
        else:
            decoding_token_tensor=body_token_tensor
        
    
        
        if order/whole<0.33 :
            order_token_tensor=front_token_tensor
        elif order/whole <0.66 :
            order_token_tensor=middle_token_tensor
        else:
            order_token_tensor=ending_token_tensor
        

        alpha=0.5
        beta=0.5
        if USE_ALPHA:
            if short_prev.shape[1]>500:
                short_prev=short_prev[:,-500:]
            
            previous=torch.cat((short_prev,eoprev_token_tensor,decoding_token_tensor,order_token_tensor,next_is_ending_token_tensor),1)
            previous=tokenizer.decode(previous[0],skip_special_tokens=True)
            previous=bert_tokenizer(previous,return_tensors="pt").input_ids.to(gpu_name)

            output=self.bert(previous)
            alpha=self.rogistic(output.pooler_output)
            alpha=self.sigmoid(alpha) 
            alpha=torch.mul((alpha),1/2)
            beta=0.5-alpha

        if debug:
            print("alpha :")
            print(alpha)
            print("beta : ")
            print(beta)
       
         
       
        if use_rake is False or input_ids.shape[1]+conti_prev_predictions.shape[1]+5 > 1020 or intro:
            
        #    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,order_token_tensor,by_token_tensor,whole_token_tensor),1)
            if NO_IBT is False and NO_FME is False:
                if order==whole-1:
                    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,decoding_token_tensor,order_token_tensor,next_is_ending_token_tensor),1) # 다음 문단이 ending이라는 정보를 알려준다.
                else:
                    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,decoding_token_tensor,order_token_tensor),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor),1)
            elif NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,decoding_token_tensor),1)
            elif NO_IBT:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,order_token_tensor),1)
        else:
            if NO_IBT is False and NO_FME is False:
                if order==whole-1:
                    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_keyword_prev_predictions,eoprev_token_tensor,decoding_token_tensor,order_token_tensor,next_is_ending_token_tensor),1)
                else:
                    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_keyword_prev_predictions,eoprev_token_tensor,decoding_token_tensor,order_token_tensor),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_keyword_prev_predictions,eoprev_token_tensor),1)
            elif NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_keyword_prev_predictions,eoprev_token_tensor,decoding_token_tensor),1)
            elif NO_IBT:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_keyword_prev_predictions,eoprev_token_tensor,order_token_tensor),1)
                
        input_ids=torch.cat((input_ids,decoder_input_ids),1)
        labels=torch.cat((input_ids,decoder_input_ids),1)
        if debug:
            print("after preprocessing, input ids: ")
            print(tokenizer.decode(input_ids[0],skip_special_tokens=False))
            print(tokenizer.decode(labels[0],skip_special_tokens=False))

        inputs_embeds=self.shared(input_ids)
        #print(inputs_embeds.shape)
        list_attention_mask=[[0]*input_ids.shape[1]]
        for i in range(input_ids.shape[1]):
            if input_ids[0][i]!=pad_token_id: # pad token id는 1이다. pad가 아니면 1로 해야 한다.
                list_attention_mask[0][i]=1

        attention_mask=torch.LongTensor(list_attention_mask).to(gpu_name)
        if debug:
            print("attention_mask")
            print(attention_mask)

        outputs = self.gpt(input_ids = None,inputs_embeds=inputs_embeds,attention_mask = attention_mask,labels=labels,output_hidden_states=True,memory=memory,context=cumulation,alpha=alpha,beta=beta)

        return outputs,memory
    
   def generate(self, memory,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask,labels,prev_predictions,conti_prev_predictions,conti_keyword_prev_predictions,order,whole,intro,tail,use_memory,use_cumulative,use_rake):#prompt_ids,prompt_attention):

       
        short_prev=prev_predictions # 뒤에서 사용
        prev_predictions=torch.cat((torch.LongTensor([[tokenizer.pad_token_id]*(1024-prev_predictions.shape[1])]).to(gpu_name),prev_predictions),1)
        prev_predictions = self.shared(prev_predictions)
        #print(for_concat_prev_predictions.shape)
        if use_memory :
            memory=self.grucell(torch.squeeze(prev_predictions),torch.squeeze(memory)).unsqueeze(dim=0)
        else:
            memory=None
        if use_cumulative :
            cumulation=self.shared(conti_prev_predictions)
        else:
            cumulation=None

        if intro:
            decoding_token_tensor=intro_token_tensor
        elif tail:
            decoding_token_tensor=tail_token_tensor
        else:
            decoding_token_tensor=body_token_tensor

        if order/whole<0.33 :
            order_token_tensor=front_token_tensor
        elif order/whole <0.66 :
            order_token_tensor=middle_token_tensor
        else:
            order_token_tensor=ending_token_tensor

        alpha=0.5
        beta=0.5
        if USE_ALPHA:
            if short_prev.shape[1]>500:
                short_prev=short_prev[:,-500:]
            previous=torch.cat((short_prev,eoprev_token_tensor,decoding_token_tensor,order_token_tensor,next_is_ending_token_tensor),1)
            previous=tokenizer.decode(previous[0],skip_special_tokens=True)
            previous=bert_tokenizer(previous,return_tensors="pt").input_ids.to(gpu_name)
            

            output=self.bert(previous)
            alpha=self.rogistic(output.pooler_output)
            alpha=self.sigmoid(alpha) 
            alpha=torch.mul((alpha),1/2)
            beta=0.5-alpha

        if debug:
            print("alpha :")
            print(alpha)
            print("beta : ")
            print(beta)
        
        if use_rake is False or input_ids.shape[1]+conti_prev_predictions.shape[1]+5 > 1020 or intro:
            
        #    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,order_token_tensor,by_token_tensor,whole_token_tensor),1)
            if NO_IBT is False and NO_FME is False:
                if order==whole-1:
                    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,decoding_token_tensor,order_token_tensor,next_is_ending_token_tensor),1) # 다음 문단이 ending이라는 정보를 알려준다.
                else:
                    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,decoding_token_tensor,order_token_tensor),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor),1)
            elif NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,decoding_token_tensor),1)
            elif NO_IBT:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,order_token_tensor),1)
        else:
            if NO_IBT is False and NO_FME is False:
                if order==whole-1:
                    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_keyword_prev_predictions,eoprev_token_tensor,decoding_token_tensor,order_token_tensor,next_is_ending_token_tensor),1)
                else:
                    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_keyword_prev_predictions,eoprev_token_tensor,decoding_token_tensor,order_token_tensor),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_keyword_prev_predictions,eoprev_token_tensor),1)
            elif NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_keyword_prev_predictions,eoprev_token_tensor,decoding_token_tensor),1)
            elif NO_IBT:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_keyword_prev_predictions,eoprev_token_tensor,order_token_tensor),1)
    
        input_ids=torch.cat((input_ids,decoder_input_ids),1)
        labels=torch.cat((input_ids,decoder_input_ids),1)
        if debug:
            print("after preprocessing, input ids: ")
            print(tokenizer.decode(input_ids[0],skip_special_tokens=False))
            print(tokenizer.decode(labels[0],skip_special_tokens=False))

        inputs_embeds=self.shared(input_ids)
        #print(inputs_embeds.shape)
        list_attention_mask=[[0]*input_ids.shape[1]]

        for i in range(input_ids.shape[1]):
            if input_ids[0][i]!=pad_token_id: # pad token id는 1이다. pad가 아니면 1로 해야 한다.
                list_attention_mask[0][i]=1

        #attention_mask=torch.cat((prompt_attention,attention_mask),1)
        attention_mask=torch.LongTensor(list_attention_mask).to(gpu_name)

        #attention_mask=torch.cat((prompt_attention,attention_mask),1)
        #print("concat and embedded input ids shape :")
        #print(inputs_embeds.shape)
        #print("concat attention mask shape : ")
        #print(attention_mask.shape)

        #inputs_embeds=torch.cat((prev_predictions,inputs_embeds),1)
        #attention_mask=torch.cat((torch.LongTensor([[1]*prev_predictions.shape[1]]).to(gpu_name),attention_mask),1)
        #dummy_decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(gpu_name)
        #source= tokenizer.batch_decode(input_ids,skip_special_tokens=True)
        #print("source")
        #print(source)
        return self.gpt.generate(max_length=250,memory=memory,inputs_embeds=inputs_embeds,attention_mask=attention_mask,
                #num_beams=4,
                do_sample=True,
                top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
                top_p=0.95,
                no_repeat_ngram_size=3,
                #encoder_no_repeat_ngram_size=3,
                repetition_penalty=3.5,early_stopping=True,context=cumulation,alpha=alpha,beta=beta),memory

config = AutoConfig.from_pretrained('gpt2-medium')
bert_config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")





if CONTINUOUSLY_TRAIN:
    gpt =  AutoModelForCausalLM.from_config(config).to(gpu_name) # 이후부터는 내가 finetune한 bart를 사용(밑에서 torch로 불러온다.)
    bert = AutoModel.from_config(bert_config).to(gpu_name)
else:
    gpt = AutoModelForCausalLM.from_pretrained('gpt2-medium').to(gpu_name) # 최초 학습에서는 pretrained 된 bart를 사용
    bert = AutoModel.from_pretrained("prajjwal1/bert-tiny").to(gpu_name)

gpt.resize_token_embeddings(len(tokenizer)) # 이렇게 하면 랜덤한 embedding unit이 추가가 된다.
bert.resize_token_embeddings(len(bert_tokenizer))

#bart.get_input_embeddings().requires_grad = False # embedding layer는 학습을 안한다. 얘가 변동되면 prev_predictions에 대한 표현도 계속 변하기 때문.
#생각해보니, shared에다가 init에서 복사한 embedding module만 계속 쓰는 거잖아?
model = Network(config.vocab_size, config.n_embd,gpt, bert,bert_config).to(gpu_name)

# -----------train ends, eval starts.
# f = open('rake_fme_second_level_val_results.csv','w', newline='')
# wr = csv.writer(f)
# wr.writerow(["steps","index","source","real text","generated_results"])

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-6)
#num_epochs = 3
num_training_steps = (num_epochs-1) * len(train_dataset) + len(train_dataset)-LAST_STEP

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps

)

progress_bar = tqdm(range(num_training_steps))


if CONTINUOUSLY_TRAIN:
    checkpoint= torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def trainer(LAST_STEP):
    conti_first=False
    whole_count_for_save=0
    #do_eval(0)
    model.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        
        if conti_first:
            LAST_STEP=0
        use_cumulative=USE_CUMULATIVE
        use_memory=USE_MEMORY
        
        for i, data in enumerate(train_dataset[LAST_STEP:],0):
        # get the inputs; data is a list of [inputs, labels]
            mini_running_loss=0.0
            
            input_ids,attention_mask,num_decoder_input_ids,decoder_attention_masks= (data['input_ids'],data['input_attention'],data['decoder_input_ids'],data['decoder_attention_mask'])
            
            if input_ids.shape[1] > 1020:

                print("the whole summary length is too long! skip this data.") 
                progress_bar.update(1)
                                
                continue
        
            count=0
        
            prev_predictions=data['prompt']
        
            input_ids=input_ids.to(gpu_name)
            attention_mask=attention_mask.to(gpu_name)
            

            memory = torch.zeros_like(torch.empty(1,1024,config.d_model)).to(gpu_name) # first memory.
            #cumul_prev_predictions = torch.zeros_like(torch.empty(1,1)).to(gpu_name)
            cumul_prev_predictions=[]
            keyword_prev_predictions=[]
            conti_keyword_prev_predictions=torch.zeros_like(torch.empty(1,1),dtype=torch.long)
            conti_prev_predictions=torch.zeros_like(torch.empty(1,1),dtype=torch.long)
            
        #print(prev_predictions)
            #torch.cuda.empty_cache() # manually freeing gpu memory.
            for d in num_decoder_input_ids:

                #input()
                prev_predictions=prev_predictions.to(gpu_name)
                decoder_attention_mask=decoder_attention_masks[count]
                
            #print("before concat, prompt ids shape :")
            #print(prompt_ids.shape)
                #print(decoder_attention_masks.shape) 
                dd=torch.unsqueeze(d[:-1],dim=0).to(gpu_name)
                decoder_attention_mask=torch.unsqueeze(decoder_attention_masks[count][:-1],dim=0).to(gpu_name)
            # input_ids 맨 앞에 이전 preceding context를 합친다.
                label=torch.unsqueeze(d[1:],dim=0).to(gpu_name)
            
                
                # zero the parameter gradients
                optimizer.zero_grad()
        
                if len(cumul_prev_predictions)>0:
                    conti_prev_predictions=cumul_prev_predictions[0]
                    conti_keyword_prev_predictions=keyword_prev_predictions[0]

                if use_cumulative and count>0:
                    length=len(cumul_prev_predictions)
                    #print("one step." + str(length))
                    for j in range(1,CUMUL_NUM if length>CUMUL_NUM else length):
                        #print(prev_predictions.shape)
                        if input_ids.shape[1]+(cumul_prev_predictions[j].shape[1])+conti_prev_predictions.shape[1]>1000:
                            #print("break")
                            #print(cumul_prev_predictions[j].shape)
                            break
                        conti_prev_predictions=torch.cat((conti_prev_predictions,sep_token_tensor,cumul_prev_predictions[j]),1)       
                        conti_keyword_prev_predictions=torch.cat((conti_keyword_prev_predictions,sep_token_tensor,keyword_prev_predictions[j]),1)
                
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


                count+=1
                order=count
                whole=len(num_decoder_input_ids)
                conti_prev_predictions=conti_prev_predictions.to(gpu_name)
                conti_keyword_prev_predictions=conti_keyword_prev_predictions.to(gpu_name)

                if use_memory is False:
                    memory = torch.zeros_like(torch.empty(1,1024,config.d_model)).to(gpu_name)
                outputs,memory = model(memory=memory.detach(),input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = dd,decoder_attention_mask=decoder_attention_mask,labels=label,prev_predictions=prev_predictions,
                                       conti_prev_predictions=conti_prev_predictions,conti_keyword_prev_predictions=conti_keyword_prev_predictions,order=order,whole=whole,intro=intro,tail=tail,use_cumulative=use_cumulative,use_memory=use_memory,use_rake=USE_RAKE)#prompt_ids=prompt_ids,prompt_attention=prompt_attention) # 중요! memory.detach()를 하지 않으면 매번 memory cell에 대한 gradient는 계속 이어져나가 계산되기 때문에, 두번 그래디언트 업데이트 했다고 오류 뜬다.
                

                

                loss = outputs.loss
                loss.backward()

                
                prev_predictions =  torch.argmax(outputs.logits, dim=-1)
                
                if TEACHER_FORCING_MEMORY:
                    prev_predictions = label # teacher forcing으로, memory와 cumul에 쓰이는 prev prediction은 training 과정에선 golden label 사용!
                
                if USE_FUSION is True:
                    use_cumulative=True ## fusion ver.

                if use_cumulative:
                    cumul_prev_predictions.insert(0,prev_predictions)
                
                
                if use_cumulative:
                    r.extract_keywords_from_text(tokenizer.decode(prev_predictions[0],skip_special_tokens=True))
                    top_features = r.get_ranked_phrases()
                    topK=10
                    if len(top_features)==0:
                        keyword_prev_predictions.insert(0,sep_token_tensor)
                    else:
                        top_features = clean_top_features(top_features, topK)
                        keywordsSTR = convert_keys_to_str(top_features)

                        keyword_prev_predictions.insert(0,tokenizer(keywordsSTR,return_tensors='pt').input_ids.to(gpu_name))
                    if debug:
                        print("keywords from last output:")
                        print(keywordsSTR)
                        print("shape")
                        print(tokenizer(keywordsSTR,return_tensors='pt').input_ids.shape)
                if debug:    
                    input()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # print statistics
                mini_running_loss += loss.item()
            
            running_loss +=mini_running_loss / count
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

            
        do_eval(epoch * len(train_dataset)+i)
        writer.flush()

        conti_first=True
    print('Finished Training')

    writer.flush()
    torch.save({'epoch':num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },PATH)


trainer(LAST_STEP)
writer.close()
