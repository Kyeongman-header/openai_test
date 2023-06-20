import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import AutoConfig
from dataset_consts import *
import random
from torch.utils.tensorboard import SummaryWriter
import sys
from rake_nltk import Rake
r = Rake()
writer = SummaryWriter()
from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer



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
use_mem=int(sys.argv[2]) # 1
use_cumul=int(sys.argv[3]) # 1
use_rake=int(sys.argv[4])
use_alpha=int(sys.argv[5])
use_fusion=int(sys.argv[6])
cumul_num=int(sys.argv[7]) # 3
no_ibt=int(sys.argv[8])
no_fme=int(sys.argv[9])
num_paragraph=int(sys.argv[10])
human_eval=int(sys.argv[11])
gpu_name=sys.argv[12] # cuda:0
debug = int(sys.argv[13]) # 1

if debug ==1:
    debug=True
else:
    debug=False

HUMAN_EVAL=True
if human_eval==1:
    HUMAN_EVAL=True
else:
    HUMAN_EVAL=False

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

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler,AutoTokenizer
from dataset_consts import *
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
   def __init__(self, vocab_size, d_model,bart,bert,bert_config): 
       super(Network, self).__init__() 
        
       self.shared = bart.get_input_embeddings()
       #nn.Embedding(config.vocab_size, config.d_model) 
       self.shared.requires_grad = False # 이 shared는 역할상 고정되어 있어야 한다.
       # 하지만 bart의 embedding layer는 학습을 거치면서 업데이트 된다.
       self.bart = bart
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

   def forward(self, memory,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask,labels,prev_predictions,conti_prev_predictions,conti_keyword_prev_predictions,order,whole,intro,tail,use_cumulative,use_memory,use_rake,debug):#prompt_ids,prompt_attention):
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
                

        if debug:
            print("after preprocessing, input ids: ")
            print(tokenizer.decode(input_ids[0],skip_special_tokens=False))

        inputs_embeds=self.shared(input_ids)
        #print(inputs_embeds.shape)
        list_attention_mask=[[0]*input_ids.shape[1]]
        for i in range(input_ids.shape[1]):
            if input_ids[0][i]!=1: # pad token id는 1이다. pad가 아니면 1로 해야 한다.
                list_attention_mask[0][i]=1

        attention_mask=torch.LongTensor(list_attention_mask).to(gpu_name)

        

        outputs = self.bart(input_ids = None,inputs_embeds=inputs_embeds,attention_mask = attention_mask,decoder_input_ids = decoder_input_ids,decoder_attention_mask=decoder_attention_mask,labels=labels,output_hidden_states=True,memory=memory,context=cumulation,alpha=alpha,beta=beta)

        return outputs,memory
    
   def generate(self, memory,input_ids,attention_mask,prev_predictions,conti_prev_predictions,conti_keyword_prev_predictions,order,whole,intro,tail,use_memory,use_cumulative,use_rake,debug,decoder_input_ids=None,decoder_attention_mask=None,labels=None,):#prompt_ids,prompt_attention):

       
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
    
        if debug:
            print("after preprocessing, input ids: ")
            print(tokenizer.decode(input_ids[0],skip_special_tokens=False))


        inputs_embeds=self.shared(input_ids)
        #print(inputs_embeds.shape)
        list_attention_mask=[[0]*input_ids.shape[1]]

        for i in range(input_ids.shape[1]):
            if input_ids[0][i]!=1: # pad token id는 1이다. pad가 아니면 1로 해야 한다.
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
        return self.bart.generate(max_length=250,memory=memory,inputs_embeds=inputs_embeds,attention_mask=attention_mask,
                #num_beams=4,
                do_sample=True,
                top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
                top_p=0.95,
                no_repeat_ngram_size=3,
                #encoder_no_repeat_ngram_size=3,
                repetition_penalty=3.5,early_stopping=True,context=cumulation,alpha=alpha,beta=beta),memory



config = AutoConfig.from_pretrained('facebook/bart-base')
bart =  AutoModelForSeq2SeqLM.from_config(config).to(gpu_name)
bart.resize_token_embeddings(len(tokenizer))
bert_config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
bert = AutoModel.from_config(bert_config).to(gpu_name)
bert.resize_token_embeddings(len(bert_tokenizer))

model = Network(config.vocab_size, config.d_model,bart, bert,bert_config).to(gpu_name)




PATH = './second_level/'+save_dir+'.tar'

checkpoint= torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

with open("pickle_data/"+"test"+"/level_2.pickle","rb") as fi:
        test_dataset = pickle.load(fi)
if HUMAN_EVAL is True:
    print("Please enter your name or survey's name.")
    name=input()
if HUMAN_EVAL is True:        
    f = open('length_control_'+str(num_paragraph)+'_'+save_dir+'_'+name+"_human-eval"+'csv','w', newline='')
else:
    f = open('length_control_'+str(num_paragraph)+'_'+save_dir+"_automatic-eval_"+'csv','w', newline='')

wr = csv.writer(f)
wr.writerow(["steps","index","source","not real text","generated_results"])


def make_any_req(NUM_PARAGRAPH=5,index=0,keywords="keywords text.",prompt="prompt"):
    
    input=tokenizer(keywords,return_tensors="pt")
    input_ids=input['input_ids']
    attention_mask=input['attention_mask']
    memory = torch.zeros_like(torch.empty(1,1024,config.d_model)).to(gpu_name) # first memory.
    prev_predictions=tokenizer(prompt,return_tensors="pt")['input_ids']
    cumul_prev_predictions=[]
    conti_prev_predictions=torch.zeros_like(torch.empty(1,1),dtype=torch.long)
    keyword_prev_predictions=[]
    conti_keyword_prev_predictions=torch.zeros_like(torch.empty(1,1),dtype=torch.long)
    whole=NUM_PARAGRAPH
    whole_predictions=[]
    model.eval()
    count=0

    use_cumulative=USE_CUMULATIVE
    use_memory=USE_MEMORY
    

    for o in range(1,NUM_PARAGRAPH+1):
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
        
        if count==NUM_PARAGRAPH-1:
            tail=True
            intro=False
            if USE_FUSION is True:
                use_memory=False
                use_cumulative=True
                ## fusion ver.

        count+=1
        order=count
        whole=NUM_PARAGRAPH
        conti_prev_predictions=conti_prev_predictions.to(gpu_name)
        conti_keyword_prev_predictions=conti_keyword_prev_predictions.to(gpu_name)
        if use_memory is False:
            memory = torch.zeros_like(torch.empty(1,1024,config.d_model)).to(gpu_name)
                
        outputs,memory=model.generate(memory=memory.detach(),
                                        input_ids = input_ids,attention_mask = attention_mask,
                                        prev_predictions=prev_predictions,conti_prev_predictions=conti_prev_predictions,
                                        conti_keyword_prev_predictions=conti_keyword_prev_predictions,order=order,whole=whole,intro=intro,tail=tail,
                                        use_cumulative=use_cumulative,use_memory=use_memory,use_rake=USE_RAKE,debug=debug)#prompt_ids=prompt_ids,prompt_attention=prompt_attention)
            
        prev_predictions = outputs # 이렇게 만들면 outputs에 id가 나오는 모양임.
        
        predictions=tokenizer.batch_decode(outputs,skip_special_tokens=True)
        whole_predictions.append(predictions)
        
        wr.writerow(["lengthcontrol",str(index),tokenizer.batch_decode(input_ids,skip_special_tokens=True),predictions,predictions]) # real 자리에 그냥 prediction 넣었다
        index+=1

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
            print("-----------")
            print("predictions")
            print(predictions)
            input()
    
    print("the result is :")
    print(whole_predictions)
    return whole_predictions
import random
from googletrans import Translator
import numpy as np

translator = Translator()
scores_a=[]
scores_b=[]
scores_c=[]


if HUMAN_EVAL is True:
    _range=random.sample(test_dataset,10)
else:
    _range=test_dataset[0:10]

for z,data in enumerate(_range):
    if HUMAN_EVAL:
        print("the " + str(z) + "th survey.")
        print("The keyword : ")
    input_ids,_,_,_,prompt = (data['input_ids'],data['input_attention'],data['decoder_input_ids'],data['decoder_attention_mask'],data['prompt'])
    keywords=tokenizer.batch_decode(input_ids,skip_special_tokens=True)
    prompt=tokenizer.batch_decode(prompt,skip_special_tokens=True)
    if HUMAN_EVAL or debug:
        print(keywords[0])
        print("Prompt : ")
        print(prompt[0])
    
    if HUMAN_EVAL is False:
        predictions=make_any_req(NUM_PARAGRAPH=num_paragraph,keywords=keywords[0],prompt=prompt[0])
        if debug :
            print("The result : ")
            print(" ".join(predictions))
            print("Korean Translation : (구글 번역기의 번역 결과임을 감안하여 평가해 주세요.)" )
            translated = translator.translate(" ".join(predictions), dest='ko')
            input()
            

    elif HUMAN_EVAL:
        predictions=make_any_req(NUM_PARAGRAPH=num_paragraph,keywords=keywords[0],prompt=prompt[0])
        print("The result : ")
        print(" ".join(predictions))
        print("Korean Translation : (구글 번역기의 번역 결과임을 감안하여 평가해 주세요.)" )
        translated = translator.translate(" ".join(predictions), dest='ko')
        print(translated.text)
        print()
        while True:
            print("Question 1. 이 글은 주제의 통일성이 마치 사람이 쓴 것과 같다.")
            print("1. 매우 아니다. 2. 아니다. 3. 보통이다. 4. 그렇다 5. 매우 그렇다.")
            print("Answer the number.",end=" ")
            a=input()
            if a.isdigit() and int(a)<=5 and int(a)>0:
                break
            else:
                print("You answerd wrong case => " + a)
                print("Please answer the question again.")

        while True:
            print("Question 2. 이 글은 한편의 글로써 완결성이 마치 사람이 쓴 것과 같다.")
            print("1. 매우 아니다. 2. 아니다. 3. 보통이다. 4. 그렇다. 5. 매우 그렇다.")
            print("Answer the number.",end=" ")
            b=input()
            if b.isdigit() and int(b)<=5 and int(b)>0:
                break
            else:
                print("You answerd wrong case => " + b)
                print("Please answer the question again.")
        
        while True:
            print("Question 3. 이 글은 사람이 쓴 것 같다.")
            print("1. 매우 아니다. 2. 아니다. 3. 보통이다. 4. 그렇다. 5. 매우 그렇다.")
            print("Answer the number.",end=" ")
            c=input()
            if c.isdigit() and int(c)<=5 and int(c)>0:
                break
            else:
                print("You answerd wrong case => " + c)
                print("Please answer the question again.")
        
        
        scores_a.append(int(a))
        scores_b.append(int(b))
        scores_c.append(int(c))
        if debug:
            print("a : " + a)
            print("b : " + b)
            print("c : " + c)
        print("If you want to stop this survey, please enter the '0'. If you enter whatever else including the 'enter', the survey will be keep going.")
        stop=input()
        print("You entered " + stop)
        if stop=='0':
            break
            
if HUMAN_EVAL:
    scores_a=np.array(scores_a)
    scores_b=np.array(scores_b)
    scores_c=np.array(scores_c)
    print("총 " + str(scores_a.size) + "개의 문항을 답했습니다.")
    print("1번 문항 평균, 분산. ")
    print(np.mean(scores_a))
    print(np.var(scores_a))
    print("2번 문항 평균, 분산. ")
    print(np.mean(scores_b))
    print(np.var(scores_b))
    print("3번 문항 평균, 분산. ")
    print(np.mean(scores_c))
    print(np.var(scores_c))

    import os
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error Creating directory. ' + directory)
    createFolder('LengthControlEvaluate')
    createFolder('LengthControlEvaluate/'+ save_dir)
    createFolder('LengthControlEvaluate/'+ save_dir + '/' + name + "_"+str(num_paragraph))

    np.save('HumanEvaluate/'+save_dir+'/'+name+"_"+str(num_paragraph)+"/f_scores_a", scores_a)
    np.save('HumanEvaluate/'+save_dir+'/'+name+"_"+str(num_paragraph)+"/f_scores_b", scores_b)
    np.save('HumanEvaluate/'+save_dir+'/'+name+"_"+str(num_paragraph)+"/f_scores_c", scores_c)
