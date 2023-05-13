import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import AutoConfig
from dataset_consts import *
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

conti=1 # 0

use_mem=int(sys.argv[3]) # 1
use_cumul=int(sys.argv[4]) # 1

cumul_num=int(sys.argv[5]) # 3

no_ibt=int(sys.argv[6])
no_fme=int(sys.argv[7])

eval_self_bleu=int(sys.argv[8])

dataset_dir=sys.argv[9]

gpu_name=sys.argv[10] # cuda:0
debug = int(sys.argv[11]) # 1

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

EVAL_SELF_BLEU=False
if eval_self_bleu==1:
    EVAL_SELF_BLEU=True
else:
    EVAL_SELF_BLEU=False

print('save_dir : ' + save_dir)
print('log_dir : ' + log_dir)
print('use_cumulative : ')
print(USE_CUMULATIVE)
print('cumul num :')
print(cumul_num)
print('use_mem : ')
print(USE_MEMORY)
print('continuously : ')
print(CONTINUOUSLY_TRAIN)

print("no_ibt : ")
print(NO_IBT)
print("no_fme : ")
print(NO_FME)
print("dataset dir : ")
print(dataset_dir)
print("self bleu eval : ")
print(EVAL_SELF_BLEU)
print("gpu or cpu num :")
print(gpu_name)
print("debug : ")
print(debug)

# num_added_toks = tokenizer.add_tokens(["<plot>","</plot>","<prev>","</prev>","<by>","<sep>"],special_tokens=True)
num_added_toks = tokenizer.add_tokens(["<plot>","</plot>","<prev>","</prev>","<i>","<b>","<t>","<f>","<m>","<e>","[SEP]"],special_tokens=True)
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

with open("pickle_data/"+"test_"+dataset_dir+"/level_2.pickle","rb") as fi:
        test_dataset = pickle.load(fi)

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module): 
   def __init__(self, vocab_size, d_model,bart): 
       super(Network, self).__init__() 
        
       self.shared = bart.get_input_embeddings()
       #nn.Embedding(config.vocab_size, config.d_model) 
       self.shared.requires_grad = False # 이 shared는 역할상 고정되어 있어야 한다.
       # 하지만 bart의 embedding layer는 학습을 거치면서 업데이트 된다.
       self.bart = bart
       self.grucell=nn.GRUCell(d_model,d_model).to(gpu_name)
       """
       self.wHr = nn.Linear(d_model,d_model).to(gpu_name)
       self.wMr = nn.Linear(d_model,d_model).to(gpu_name)
       self.wHz = nn.Linear(d_model,d_model).to(gpu_name)
       self.wMz = nn.Linear(d_model,d_model).to(gpu_name)
       self.wHn = nn.Linear(d_model,d_model).to(gpu_name)
       self.wMn = nn.Linear(d_model,d_model).to(gpu_name)
       """

   def forward(self, memory,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask,labels,prev_predictions,conti_prev_predictions,order,whole,intro,tail):#prompt_ids,prompt_attention):
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

        prev_predictions=torch.cat((torch.LongTensor([[tokenizer.pad_token_id]*(1024-prev_predictions.shape[1])]).to(gpu_name),prev_predictions),1)
        
        prev_predictions = self.shared(prev_predictions)
       #print(prev_predictions.shape)
        
        if USE_MEMORY :
           memory=self.grucell(torch.squeeze(prev_predictions),torch.squeeze(memory)).unsqueeze(dim=0)
        else:
           memory=memory
       #print(memory.shape)

       #print("embedded prev prediction : ")
       #print(prev_predictions.shape)
       
       # prev_predictions = torch.mean(prev_predictions,dim=-2,keepdim=True)

       # prev_predictions을 mean때리지 않고 마치 decoder에 encoder output을 prompt로 넣듯이 똑같이 해보자. 

       #print("embedded prev prediction : ")
       #print(prev_predictions.shape)
       

       #print("before concat, prompt ids shape :")
       #print(prompt_ids.shape)
       #print("before concat, input ids shape :")
       #print(input_ids.shape)

       #input_ids=torch.cat((prompt_ids,input_ids),1)
       
       #print("input id shape")
       #print(input_ids.shape)
    #    order_token_tensor=torch.LongTensor([[order]]).to(gpu_name)
    #    whole_token_tensor=torch.LongTensor([[whole]]).to(gpu_name)
        

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
         
       
        if USE_CUMULATIVE is False or input_ids.shape[1]+conti_prev_predictions.shape[1]+5 > 1020 or intro:
            
        #    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,order_token_tensor,by_token_tensor,whole_token_tensor),1)
            if NO_IBT is False and NO_FME is False:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,decoding_token_tensor,order_token_tensor),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor),1)
            elif NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,decoding_token_tensor),1)
            elif NO_IBT:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,order_token_tensor),1)
        else:
            if NO_IBT is False and NO_FME is False:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_prev_predictions,eoprev_token_tensor,decoding_token_tensor,order_token_tensor,),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_prev_predictions,eoprev_token_tensor),1)
            elif NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_prev_predictions,eoprev_token_tensor,decoding_token_tensor),1)
            elif NO_IBT:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_prev_predictions,eoprev_token_tensor,order_token_tensor),1)
                

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

        #print("attention mask")
        #print(attention_mask.shape)
        #print("decoder attention mask")
        #print(decoder_attention_mask.shape)
        #print("decoder input ids")
        #print(decoder_input_ids.shape)

        #print("concat and embedded input ids shape :")
        #print(inputs_embeds.shape)
        #print("concat attention mask shape : ")
        #print(attention_mask.shape)

        #inputs_embeds=torch.cat((prev_predictions,inputs_embeds),1)
        #attention_mask=torch.cat((torch.LongTensor([[1]*prev_predictions.shape[1]]).to(gpu_name),attention_mask),1)

        #print("prev concat input embeds shape : ")
        #print(inputs_embeds.shape)
        #print("prev concat attention mask shape : ")
        #print(attention_mask.shape)
        #print("decoder intput ids")
        #print(decoder_input_ids.shape)
        #print("decoder attention mask")
        #print(decoder_attention_mask.shape)

        outputs = self.bart(input_ids = None,inputs_embeds=inputs_embeds,attention_mask = attention_mask,decoder_input_ids = decoder_input_ids,decoder_attention_mask=decoder_attention_mask,labels=labels,output_hidden_states=True,memory=memory)

        return outputs,memory
    
   def generate(self, memory,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask,labels,prev_predictions,conti_prev_predictions,order,whole,intro,tail):#prompt_ids,prompt_attention):

        prev_predictions=torch.cat((torch.LongTensor([[tokenizer.pad_token_id]*(1024-prev_predictions.shape[1])]).to(gpu_name),prev_predictions),1)
        prev_predictions = self.shared(prev_predictions)
        #print(for_concat_prev_predictions.shape)
        if USE_MEMORY :
            memory=self.grucell(torch.squeeze(prev_predictions),torch.squeeze(memory)).unsqueeze(dim=0)
        else:
            memory=memory
       
       
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


        if USE_CUMULATIVE is False or input_ids.shape[1]+conti_prev_predictions.shape[1]+5 > 1020 or intro:
            # print("no previous decoder output used because of too long summary.")

        #    input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,order_token_tensor,by_token_tensor,whole_token_tensor),1)
            if NO_IBT is False and NO_FME is False:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,decoding_token_tensor,order_token_tensor),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor),1)
            elif NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,decoding_token_tensor),1)
            elif NO_IBT:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,order_token_tensor),1)

        else:
            if NO_IBT is False and NO_FME is False:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_prev_predictions,eoprev_token_tensor,decoding_token_tensor,order_token_tensor,),1)
            elif NO_IBT and NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_prev_predictions,eoprev_token_tensor),1)
            elif NO_FME:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_prev_predictions,eoprev_token_tensor,decoding_token_tensor),1)
            elif NO_IBT:
                input_ids=torch.cat((soplot_token_tensor,input_ids,eoplot_token_tensor,soprev_token_tensor,conti_prev_predictions,eoprev_token_tensor,order_token_tensor),1)
        #print("input id shape")
        #print(input_ids.shape)

        inputs_embeds=self.shared(input_ids)
        #print(inputs_embeds.shape)
        list_attention_mask=[[0]*input_ids.shape[1]]

        for i in range(input_ids.shape[1]):
            if input_ids[0][i]!=1: # pad token id는 1이다. pad가 아니면 1로 해야 한다.
                list_attention_mask[0][i]=1

        #attention_mask=torch.cat((prompt_attention,attention_mask),1)
        attention_mask=torch.LongTensor(list_attention_mask).to(gpu_name)

        return self.bart.generate(max_length=250,memory=memory,inputs_embeds=inputs_embeds,attention_mask=attention_mask,
                #num_beams=4,
                do_sample=True,
                top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
                top_p=0.95,
                no_repeat_ngram_size=3,
                #encoder_no_repeat_ngram_size=3,
                repetition_penalty=3.5,early_stopping=True),memory

config = AutoConfig.from_pretrained('facebook/bart-base')
if CONTINUOUSLY_TRAIN:
    bart =  AutoModelForSeq2SeqLM.from_config(config).to(gpu_name) # 이후부터는 내가 finetune한 bart를 사용(밑에서 torch로 불러온다.)
else:
    bart = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base').to(gpu_name) # 최초 학습에서는 pretrained 된 bart를 사용
bart.resize_token_embeddings(len(tokenizer)) # 이렇게 하면 랜덤한 embedding unit이 추가가 된다.
#bart.get_input_embeddings().requires_grad = False # embedding layer는 학습을 안한다. 얘가 변동되면 prev_predictions에 대한 표현도 계속 변하기 때문.
#생각해보니, shared에다가 init에서 복사한 embedding module만 계속 쓰는 거잖아?
model = Network(config.vocab_size, config.d_model,bart).to(gpu_name)
if CONTINUOUSLY_TRAIN:
    checkpoint= torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("#######load successfully.##############")
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def do_eval(steps):
    f = open(log_dir+'_generations_outputs.csv','w', newline='')
    wr = csv.writer(f)
    wr.writerow(["steps","index","source","real text","generated_results"])
    index=0
    bleu_score_bi=0
    bleu_score_tri=0
    bleu_score_four=0
    bleu_score_fif=0
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
    met_result=0
    whole_nlls = []

    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.), 'fourgram' : (1/4.,1/4.,1/4.,1/4.), 'fifthgram' : (1/5.,1/5.,1/5.,1/5.,1/5.)}


    model.eval()

    for data in tqdm(test_dataset[0:1000]):
        input_ids,attention_mask,num_decoder_input_ids,decoder_attention_masks = (data['input_ids'],data['input_attention'],data['decoder_input_ids'],data['decoder_attention_mask'])
        
        if input_ids.shape[1] > 1020:
            print("the whole summary length is too long! skip this data.")
            continue

        count=0

        prev_predictions=data['prompt']

        input_ids=input_ids.to(gpu_name)
        attention_mask=attention_mask.to(gpu_name)
        memory = torch.zeros_like(torch.empty(1,1024,config.d_model)).to(gpu_name) # first memory.
        #print(prev_predictions)
        cumul_prev_predictions=[]
        conti_prev_predictions=torch.zeros_like(torch.empty(1,1),dtype=torch.long)
        one_label=[]
        one_prediction=[]
        _labels_len=0
        _predictions_len=0
        nlls=[]

        for d in num_decoder_input_ids:
        
            prev_predictions=prev_predictions.to(gpu_name)
            
            
            ex_d=torch.unsqueeze(d[:-1],dim=0).to(gpu_name)
            decoder_attention_mask=torch.unsqueeze(decoder_attention_masks[count][:-1],dim=0).to(gpu_name)
            label=torch.unsqueeze(d[1:],dim=0).to(gpu_name)

            if len(cumul_prev_predictions)>0:
                    conti_prev_predictions=cumul_prev_predictions[0]

            if USE_CUMULATIVE and count>0:
                length=len(cumul_prev_predictions)
                #print("one step." + str(length))
                for j in range(1,CUMUL_NUM if length>CUMUL_NUM else length):
                    #print(prev_predictions.shape)
                    if input_ids.shape[1]+(cumul_prev_predictions[j].shape[1])+conti_prev_predictions.shape[1]>1000:
                        #print("break")
                        #print(cumul_prev_predictions[j].shape)
                        break
                    conti_prev_predictions=torch.cat((conti_prev_predictions,sep_token_tensor,cumul_prev_predictions[j]),1)  
            
            
            intro=False
            tail=False
            if count==0:
                intro=True
            if count==len(num_decoder_input_ids)-1:
                tail=True
                intro=False
            count+=1
            order=count
            whole=len(num_decoder_input_ids)
            conti_prev_predictions=conti_prev_predictions.to(gpu_name)
            if USE_MEMORY is False:
                memory = torch.zeros_like(torch.empty(1,1024,config.d_model)).to(gpu_name)
            
            _memory=memory
            outputs,memory=model.generate(memory=memory.detach(),input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = ex_d,decoder_attention_mask=decoder_attention_mask,labels=label,prev_predictions=prev_predictions,conti_prev_predictions=conti_prev_predictions,order=order,whole=whole,intro=intro,tail=tail)#prompt_ids=prompt_ids,prompt_attention=prompt_attention)
            with torch.no_grad():
                dd = tokenizer.batch_decode(ex_d,skip_special_tokens=True)
                dd = tokenizer(dd,return_tensors="pt")
                dd_attention_mask=dd['attention_mask']
                dd = dd['input_ids']
                
                ddd=dd[:,:-1].to(gpu_name)
                dd_attention_mask=dd_attention_mask[:,:-1].to(gpu_name)
                dlabel=dd[:,1:].to(gpu_name)
                
                for_perplexity,_=model(memory=_memory.detach(),input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = ddd,decoder_attention_mask=dd_attention_mask,labels=dlabel,prev_predictions=prev_predictions,
                                       conti_prev_predictions=conti_prev_predictions,order=order,whole=whole,intro=intro,tail=tail)
                neg_log_likelihood=for_perplexity.loss
            
            nlls.append(neg_log_likelihood)
            
            
            prev_predictions = outputs # 이렇게 만들면 outputs에 id가 나오는 모양임.
            
            predictions = tokenizer.batch_decode(outputs,skip_special_tokens=True)
            
            if USE_CUMULATIVE:
                #print(predictions)
                r.extract_keywords_from_text(predictions[0])
                top_features = r.get_ranked_phrases()
                topK=10
                if len(top_features)==0:
                        cumul_prev_predictions.insert(0,sep_token_tensor)
                else:
                    top_features = clean_top_features(top_features, topK)
                    keywordsSTR = convert_keys_to_str(top_features)

                    cumul_prev_predictions.insert(0,tokenizer(keywordsSTR,return_tensors='pt').input_ids.to(gpu_name))

            one_prediction.append(predictions[0])
            #whole_predictions.append(predictions[0])
            _predictions_len+=len(outputs[0])
            #print("-----------")
            #print(len(outputs[0]))
            # print("predictions")

            # print(predictions) 
            #label = tokenizer.batch_decode(label,skip_special_tokens=True)
            # print("golden label")
            # print(label)
        
            #print("decoder input")
            _labels_len+=len(dd[0])
            #print(len(dd[0]))
            ex_d = tokenizer.batch_decode(ex_d,skip_special_tokens=True)
            one_label.append(ex_d[0])
            #whole_labels.append(ex_d[0])
            
            # print(ex_d)

        #print("loss")
        #print(loss)
            wr.writerow([str(steps),str(index),tokenizer.batch_decode(input_ids,skip_special_tokens=True),ex_d,predictions])
            index+=1
            
            
            #input()
        one_prediction=' '.join(one_prediction)
        one_label=' '.join(one_label)
        if len(one_label)==0:
            print("something wrong. label is not exist.")
            print("error set : ")
            print("source : ")
            print(tokenizer.batch_decode(input_ids,skip_special_tokens=True))
            print("predictions : ")
            print(one_prediction)
            print("label : ")
            print(one_label)
            continue
        whole_labels.append(one_label)
        whole_labels_len+=_labels_len
        whole_predictions.append(one_prediction)
        whole_predictions_len+=_predictions_len
        whole_nlls+=nlls
        metric.add_batch(predictions=[one_prediction], references=[one_label])
        whole_num+=1
        #bleu = BLEU([one_label],weights)
        #bleu_score=bleu.get_score([one_prediction])
        bleu_score=_bleu.compute(predictions=[one_prediction],references=[one_label],max_order=5)
        bleu_score_bi+=bleu_score['precisions'][1]
        bleu_score_tri+=bleu_score['precisions'][2]
        bleu_score_four+=bleu_score['precisions'][3]
        bleu_score_fif+=bleu_score['precisions'][4]
        met_result += meteor.compute(predictions=[one_prediction], references=[one_label])["meteor"]

    result=metric.compute()
    print(result)
    whole_nlls=torch.stack(whole_nlls)
    nlls_mean=torch.mean(whole_nlls)
    #print(nlls)
    ppl=torch.exp(nlls_mean)
    print("ppl is : " + str(ppl.item()))
    bleu_score_bi=bleu_score_bi/whole_num
    bleu_score_tri=bleu_score_tri/whole_num
    bleu_score_four=bleu_score_four/whole_num
    bleu_score_fif=bleu_score_fif/whole_num

    print("bleu_score_bi : " + str(bleu_score_bi) + " bleu_score_tri : " + str(bleu_score_tri) + " bleu_score_four : " + str(bleu_score_four) + " bleu_score_fif : " + str(bleu_score_fif))
    
    met_result=met_result/whole_num
    print("len of sample generation : " + str(whole_num))
    print("meteor : " + str(met_result))
    whole_predictions_len=whole_predictions_len/whole_num
    whole_labels_len=(whole_labels_len/whole_num)
    print("avg prediction len : " + str(whole_predictions_len))
    print("avg reference len : " + str(whole_labels_len))
    #self_bleu = SelfBLEU(whole_predictions, weights).get_score()
    #real_self_bleu = SelfBLEU(whole_labels, weights).get_score()

    if EVAL_SELF_BLEU:
        for j in range(len(whole_predictions)):
            except_whole_predictions=whole_predictions[0:j]+whole_predictions[j+1:]
            
            #self_bleu=BLEU(except_whole_predictions,weights).get_score([whole_predictions[j]])
            self_bleu=_bleu.compute(predictions=[whole_predictions[j]],references=[except_whole_predictions],max_order=5)
            self_bleu_one+=self_bleu['precisions'][0]
            self_bleu_bi+=self_bleu['precisions'][1]
            self_bleu_tri+=self_bleu['precisions'][2]
            self_bleu_four+=self_bleu['precisions'][3]
            self_bleu_fif+=self_bleu['precisions'][4]
        
        self_num=0
        for j in range(len(whole_labels)):
            except_whole_labels=whole_labels[0:j]+whole_labels[j+1:]
            real_self_bleu=_bleu.compute(predictions=[whole_labels[j]],references=[except_whole_labels],max_order=5)
            r_self_bleu_one+=real_self_bleu['precisions'][0]
            r_self_bleu_bi+=real_self_bleu['precisions'][1]
            r_self_bleu_tri+=real_self_bleu['precisions'][2]
            r_self_bleu_four+=real_self_bleu['precisions'][3]
            r_self_bleu_fif+=real_self_bleu['precisions'][4]
                
            #print(_real_self_bleu)
            self_num+=1
        
        self_bleu_one=self_bleu_one/len(whole_predictions)
        self_bleu_bi=self_bleu_bi/len(whole_predictions)
        self_bleu_tri=self_bleu_tri/len(whole_predictions)
        self_bleu_four=self_bleu_four/len(whole_predictions)
        self_bleu_fif=self_bleu_fif/len(whole_predictions)
        r_self_bleu_one=r_self_bleu_one/(self_num)
        r_self_bleu_bi=r_self_bleu_bi/(self_num)
        r_self_bleu_tri=r_self_bleu_tri/(self_num)
        r_self_bleu_four=r_self_bleu_four/(self_num)
        r_self_bleu_fif=r_self_bleu_fif/(self_num)

        
        print("self_bleu one : " + str(self_bleu_one))
        print("self_bleu bi : " + str(self_bleu_bi))
        print("self_bleu tri : " + str(self_bleu_tri))
        print("self_bleu four : " + str(self_bleu_four))
        print("self_bleu fif : " + str(self_bleu_fif))
        
        print("real self_bleu one : " + str(r_self_bleu_one))
        print("real self_bleu bi : " + str(r_self_bleu_bi))
        print("real self_bleu tri : " + str(r_self_bleu_tri))
        print("real self_bleu four : " + str(r_self_bleu_four))
        print("real self_bleu fif : " + str(r_self_bleu_fif))
    

    writer.add_scalar("rouge1/eval", result['rouge1'], steps)
    writer.add_scalar("rouge2/eval", result['rouge2'], steps)
    writer.add_scalar("rougeL/eval", result['rougeL'], steps)
    writer.add_scalar("rougeLsum/eval", result['rougeLsum'], steps)
    writer.add_scalar("bleu bi/eval", bleu_score_bi, steps)
    writer.add_scalar("bleu tri/eval", bleu_score_tri, steps)
    writer.add_scalar("bleu four/eval", bleu_score_four, steps)
    writer.add_scalar("bleu fif/eval", bleu_score_fif, steps)
    writer.add_scalar("self bleu bi/eval", self_bleu_bi, steps)
    writer.add_scalar("self bleu tri/eval", self_bleu_tri, steps)
    writer.add_scalar("self bleu four/eval", self_bleu_four, steps)
    writer.add_scalar("self bleu fif/eval", self_bleu_fif, steps)
    writer.add_scalar("real_self bleu bi/eval", r_self_bleu_bi, steps)
    writer.add_scalar("real_self bleu tri/eval", r_self_bleu_tri, steps)
    writer.add_scalar("real_self bleu four/eval", r_self_bleu_four, steps)
    writer.add_scalar("real_self bleu fif/eval", r_self_bleu_fif, steps)
    writer.add_scalar("meteor",met_result,steps)
    writer.add_scalar("predictions avg len",whole_predictions_len,steps)
    writer.add_scalar("references avg len",whole_labels_len,steps)
    writer.add_scalar("ppl",ppl.item(),steps)

do_eval(0)
