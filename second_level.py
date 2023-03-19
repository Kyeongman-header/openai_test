import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import AutoConfig
from dataset_consts import *
import random
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer

import evaluate

metric = evaluate.load("rouge")
meteor=evaluate.load("meteor")

TRAIN_RANGE=25000
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error Creating directory. ' + directory)

createFolder('second_level')
PATH = './second_level/'+'all_2.tar'

CONTINUOUSLY_TRAIN=False
USE_MEMORY=True
USE_CUMULATIVE=True
CUMUL_NUM=3

# train_total_target=last_target[:TRAIN_RANGE]
# train_total_source=total_target[:TRAIN_RANGE]
# train_total_prompt=total_source[:TRAIN_RANGE]
# val_total_target=last_target[TRAIN_RANGE:]
# val_total_source=total_target[TRAIN_RANGE:]
# val_total_prompt=total_source[TRAIN_RANGE:]

# train_dataset=return_dataset_2(train_total_target,train_total_source,train_total_prompt)
# valid_dataset=return_dataset_2(val_total_target,val_total_source,val_total_prompt)


#with open("pickle_data/"+"train"+"/level_2.pickle","rb") as fi:
with open("pickle_data/"+"cnn_dailymail/train"+"/level_2.pickle","rb") as fi:
        train_dataset = pickle.load(fi)
with open("pickle_data/"+"cnn_dailymail/validation"+"/level_2.pickle","rb") as fi:
        valid_dataset = pickle.load(fi)



import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler






class Network(nn.Module): 
   def __init__(self, vocab_size, d_model,bart): 
       super(Network, self).__init__() 
        
       self.shared = bart.get_input_embeddings()
       #nn.Embedding(config.vocab_size, config.d_model) 
       self.bart = bart
       self.grucell=nn.GRUCell(d_model,d_model).to('cuda:1')
       """
       self.wHr = nn.Linear(d_model,d_model).to('cuda:1')
       self.wMr = nn.Linear(d_model,d_model).to('cuda:1')
       self.wHz = nn.Linear(d_model,d_model).to('cuda:1')
       self.wMz = nn.Linear(d_model,d_model).to('cuda:1')
       self.wHn = nn.Linear(d_model,d_model).to('cuda:1')
       self.wMn = nn.Linear(d_model,d_model).to('cuda:1')
       """

   def forward(self, memory,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask,labels,output_hidden_states,prev_predictions,prompt_ids,prompt_attention):
       #memory states update.
       
       for_concat_prev_predictions=prev_predictions
       #print("prev_predictions shape:")
       #print(for_concat_prev_predictions.shape)
       prev_predictions=torch.cat((torch.LongTensor([[tokenizer.pad_token_id]*(1024-prev_predictions.shape[1])]).to('cuda:1'),prev_predictions),1)
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

       input_ids=torch.cat((prompt_ids,input_ids),1)
       input_ids=torch.cat((input_ids,for_concat_prev_predictions),1)
       #print("input id shape")
       #print(input_ids)
       
       inputs_embeds=self.shared(input_ids)
       #print(inputs_embeds.shape)
       list_attention_mask=[[0]*input_ids.shape[1]]
       for i in range(input_ids.shape[1]):
           if input_ids[0][i]!=1: # pad token id는 1이다. pad가 아니면 1로 해야 한다.
               list_attention_mask[0][i]=1

       #attention_mask=torch.cat((prompt_attention,attention_mask),1)
       attention_mask=torch.LongTensor(list_attention_mask).to('cuda:1')
       
       #print("attention mask")
       #print(attention_mask)
       #print("concat and embedded input ids shape :")
       #print(inputs_embeds.shape)
       #print("concat attention mask shape : ")
       #print(attention_mask.shape)
       
       #inputs_embeds=torch.cat((prev_predictions,inputs_embeds),1)
       #attention_mask=torch.cat((torch.LongTensor([[1]*prev_predictions.shape[1]]).to('cuda:1'),attention_mask),1)

       #print("prev concat input embeds shape : ")
       #print(inputs_embeds.shape)
       #print("prev concat attention mask shape : ")
       #print(attention_mask.shape)
       outputs = self.bart(input_ids = None,inputs_embeds=inputs_embeds,attention_mask = attention_mask,decoder_input_ids = decoder_input_ids,decoder_attention_mask=decoder_attention_mask,labels=labels,output_hidden_states=True,memory=memory)
       return outputs,memory
    
   def generate(self, memory,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask,labels,output_hidden_states,prev_predictions,prompt_ids,prompt_attention):
       
       for_concat_prev_predictions=prev_predictions
       prev_predictions=torch.cat((torch.LongTensor([[tokenizer.pad_token_id]*(1024-prev_predictions.shape[1])]).to('cuda:1'),prev_predictions),1)
       prev_predictions = self.shared(prev_predictions)
       #print(for_concat_prev_predictions.shape)
       if USE_MEMORY :
           memory=self.grucell(torch.squeeze(prev_predictions),torch.squeeze(memory)).unsqueeze(dim=0)
       else:
           memory=memory
       #print("embedded prev prediction : ")
       #print(prev_predictions.shape)

       #prev_predictions = torch.mean(prev_predictions,dim=-2,keepdim=True)

       #print("avg embedded prev prediction : ")
       #print(prev_predictions.shape)


       #print("before concat, prompt ids shape :")
       #print(prompt_ids.shape)
       #print("before concat, input ids shape :")
       #print(input_ids.shape)

       #input_ids=torch.cat((prompt_ids,input_ids),1)
       #inputs_embeds=self.shared(input_ids)
       
       input_ids=torch.cat((prompt_ids,input_ids),1)
       input_ids=torch.cat((input_ids,for_concat_prev_predictions),1)
       #print("input id shape")
       #print(input_ids.shape)

       inputs_embeds=self.shared(input_ids)
       #print(inputs_embeds.shape)
       list_attention_mask=[[0]*input_ids.shape[1]]
       
       for i in range(input_ids.shape[1]):
           if input_ids[0][i]!=1: # pad token id는 1이다. pad가 아니면 1로 해야 한다.
               list_attention_mask[0][i]=1

       #attention_mask=torch.cat((prompt_attention,attention_mask),1)
       attention_mask=torch.LongTensor(list_attention_mask).to('cuda:1')

       #attention_mask=torch.cat((prompt_attention,attention_mask),1)
       #print("concat and embedded input ids shape :")
       #print(inputs_embeds.shape)
       #print("concat attention mask shape : ")
       #print(attention_mask.shape)

       #inputs_embeds=torch.cat((prev_predictions,inputs_embeds),1)
       #attention_mask=torch.cat((torch.LongTensor([[1]*prev_predictions.shape[1]]).to('cuda:1'),attention_mask),1)
       #dummy_decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to('cuda:1')
       source= tokenizer.batch_decode(input_ids,skip_special_tokens=True)
       #print("source")
       #print(source)
       return self.bart.generate(max_length=250,memory=memory,inputs_embeds=inputs_embeds,attention_mask=attention_mask,num_beams=4,
               no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=3,
                repetition_penalty=3.5,early_stopping=True),memory

config = AutoConfig.from_pretrained('facebook/bart-base')
if CONTINUOUSLY_TRAIN:
    bart =  AutoModelForSeq2SeqLM.from_config(config).to('cuda:1') # 이후부터는 내가 finetune한 bart를 사용(밑에서 torch로 불러온다.)
else:
    bart = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base').to('cuda:1') # 최초 학습에서는 pretrained 된 bart를 사용

bart.get_input_embeddings().requires_grad = False # embedding layer는 학습을 안한다. 얘가 변동되면 prev_predictions에 대한 표현도 계속 변하기 때문.

model = Network(config.vocab_size, config.d_model,bart).to('cuda:1')

# -----------train ends, eval starts.
f = open('second_level_val_results.csv','w', newline='')
wr = csv.writer(f)
wr.writerow(["steps","index","source","real text","generated_results"])

def do_eval(steps):
    #f = open('second_level_val_results.csv','w', newline='')
    #wr = csv.writer(f)
    #wr.writerow(["steps","index","source","real text","generated_results"])
    index=0
    bleu_score_bi=0
    bleu_score_tri=0
    bleu_score_four=0
    bleu_score_fif=0
    self_bleu_bi=0
    self_bleu_tri=0
    self_bleu_four=0
    self_bleu_fif=0
    r_self_bleu_bi=0
    r_self_bleu_tri=0
    r_self_bleu_four=0
    r_self_bleu_fif=0
    whole_num=0
    whole_predictions=[]
    whole_labels=[]
    met_result=0

    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.), 'fourgram' : (1/4.,1/4.,1/4.,1/4.), 'fifthgram' : (1/5.,1/5.,1/5.,1/5.,1/5.)}


    model.eval()

    for data in tqdm(random.sample(valid_dataset,128)):
        input_ids,attention_mask,num_decoder_input_ids,decoder_attention_masks = (data['input_ids'],data['input_attention'],data['decoder_input_ids'],data['decoder_attention_mask'])
    

        count=0

        prev_predictions=data['prompt']

        input_ids=input_ids.to('cuda:1')
        attention_mask=attention_mask.to('cuda:1')
        memory = torch.zeros_like(torch.empty(1,1024,config.d_model)).to('cuda:1') # first memory.
        #print(prev_predictions)
        cumul_prev_predictions=[]
        for d in num_decoder_input_ids:
        
        
            prev_predictions=prev_predictions.to('cuda:1')
        
            word=""
            if count==0:
                word=str(count+1) + "st"
            elif count==1:
                word=str(count+1) + "nd"
            elif count==2:
                word=str(count+1) + "rd"
            elif count>2 and count!=len(num_decoder_input_ids)-1:
                word=str(count+1) + "th"
            else:
                word='last'
        
            prompt="MAKE A " + word + " PART OF THE ENTIRE ARTICLE. The plot : "
            
            prompt=tokenizer(prompt,return_tensors="pt")
            prompt_attention=prompt.attention_mask.to('cuda:1')
            prompt_ids=prompt.input_ids.to('cuda:1')

            #print("before concat, prompt ids shape :")
            #print(prompt_ids.shape)

        #print(d)
        #ex_d=torch.unsqueeze(d[:-1],dim=0).to('cuda:1')
        #decoder_attention_mask=torch.unsqueeze(decoder_attention_masks[count-1][:-1],dim=0).to('cuda:1')
            # input_ids 맨 앞에 이전 preceding context를 합친다.
        #label=torch.unsqueeze(d[1:],dim=0).to('cuda:1')
            ex_d=torch.unsqueeze(d,dim=0).to('cuda:1')
        
        #print(decoder_attention_masks[count-1][0])

            decoder_attention_mask=torch.unsqueeze(decoder_attention_masks[count-1],dim=0).to('cuda:1')
            label=torch.unsqueeze(d,dim=0).to('cuda:1')
            # input_ids 맨 앞에 이전 preceding context를 합친다.
            """with torch.no_grad():
            outputs,memory = model(input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = ex_d,labels=label,decoder_attention_mask=decoder_attention_mask,output_hidden_states=True,prev_predictions=prev_predictions,prompt_ids=prompt_ids,prompt_attention=prompt_attention,memory=memory.detach())
        
        

        #prev_predictions =  torch.argmax(outputs.logits, dim=-1)
        
        loss = outputs.loss
        outputs=torch.argmax(outputs.logits, dim=-1)
        predictions = tokenizer.batch_decode(outputs,)#skip_special_tokens=True)
        print("predictions")
        print(predictions)
        l = tokenizer.batch_decode(label,)#skip_special_tokens=True)
        print("golden label")
        print(l)

        print("decoder input")
        exd = tokenizer.batch_decode(ex_d,)
        print(exd)

        print("loss")
        print(loss)
            """
            if USE_CUMULATIVE and count>0:
                length=len(cumul_prev_predictions)
                
                for j in range(1,CUMUL_NUM if length>CUMUL_NUM else length):
                    if prompt_ids.shape[1]+input_ids.shape[1]+(cumul_prev_predictions[j].shape[1])+prev_predictions.shape[1]>1000:
                            break
                    prev_predictions=torch.cat((prev_predictions,cumul_prev_predictions[j]),1)

            count+=1


            outputs,memory=model.generate(memory=memory.detach(),input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = ex_d,decoder_attention_mask=decoder_attention_mask,labels=label,output_hidden_states=True,prev_predictions=prev_predictions,prompt_ids=prompt_ids,prompt_attention=prompt_attention)
            
            prev_predictions = outputs # 이렇게 만들면 outputs에 id가 나오는 모양임.
            
            predictions = tokenizer.batch_decode(outputs,skip_special_tokens=True)
            cumul_prev_predictions.insert(0,prev_predictions)
            whole_predictions.append(predictions[0])
            
            # print("predictions")
            # print(predictions) 
            #label = tokenizer.batch_decode(label,skip_special_tokens=True)
            # print("golden label")
            # print(label)
        
            #print("decoder input")
            ex_d = tokenizer.batch_decode(ex_d,skip_special_tokens=True)
            whole_labels.append(ex_d[0])
            # print(ex_d)

        #print("loss")
        #print(loss)
            wr.writerow([str(steps),str(index),tokenizer.batch_decode(input_ids,skip_special_tokens=True),ex_d,predictions])
            index+=1
            metric.add_batch(predictions=predictions, references=ex_d)
            whole_num+=1
            bleu = BLEU(ex_d,weights)
            bleu_score=bleu.get_score(predictions)
            bleu_score_bi+=bleu_score['bigram'][0]
            bleu_score_tri+=bleu_score['trigram'][0]
            bleu_score_four+=bleu_score['fourgram'][0]
            bleu_score_fif+=bleu_score['fifthgram'][0]
            met_result += meteor.compute(predictions=predictions, references=ex_d)["meteor"]
            """
            self_bleu = SelfBLEU(predictions, weights).get_score()
            self_bleu_bi+=self_bleu['bigram'][0]
            self_bleu_tri+=self_bleu['trigram'][0]
            self_bleu_four+=self_bleu['fourgram'][0]
            self_bleu_fif+=self_bleu['fifthgram'][0]
            """
            #input()
    result=metric.compute()
    print(result)
    
    bleu_score_bi=bleu_score_bi/whole_num
    bleu_score_tri=bleu_score_tri/whole_num
    bleu_score_four=bleu_score_four/whole_num
    bleu_score_fif=bleu_score_fif/whole_num

    print("bleu_score_bi : " + str(bleu_score_bi) + " bleu_score_tri : " + str(bleu_score_tri) + " bleu_score_four : " + str(bleu_score_four) + " bleu_score_fif : " + str(bleu_score_fif))
    
    met_result=met_result/whole_num
    print("len of sample generation : " + str(len(whole_predictions)))
    self_bleu = SelfBLEU(whole_predictions, weights).get_score()
    real_self_bleu = SelfBLEU(whole_labels, weights).get_score()
    
    for s in range(len(self_bleu['bigram'])):

        self_bleu_bi+=self_bleu['bigram'][s]
        self_bleu_tri+=self_bleu['trigram'][s]
        self_bleu_four+=self_bleu['fourgram'][s]
        self_bleu_fif+=self_bleu['fifthgram'][s]
        r_self_bleu_bi+=real_self_bleu['bigram'][s]
        r_self_bleu_tri+=real_self_bleu['trigram'][s]
        r_self_bleu_four+=real_self_bleu['fourgram'][s]
        r_self_bleu_fif+=real_self_bleu['fifthgram'][s]


    """
    print(len(self_bleu['bigram']))
    print(len(self_bleu['trigram']))
    print(len(self_bleu['fourgram']))
    print(len(self_bleu['fifthgram']))
    print(whole_num)
    #print(self_bleu)
    """
    print("meteor : " + str(met_result))
    
    self_bleu_bi=self_bleu_bi/len(self_bleu['bigram'])
    self_bleu_tri=self_bleu_tri/len(self_bleu['bigram'])
    self_bleu_four=self_bleu_four/len(self_bleu['bigram'])
    self_bleu_fif=self_bleu_fif/len(self_bleu['bigram'])
    r_self_bleu_bi=self_bleu_bi/(len(r_self_bleu['bigram']))
    r_self_bleu_tri=self_bleu_tri/(len(r_self_bleu['bigram']))
    r_self_bleu_four=self_bleu_four/(len(r_self_bleu['bigram']))
    r_self_bleu_fif=self_bleu_fif/(len(r_self_bleu['bigram']))

    print("self_bleu bi : " + str(self_bleu_bi))
    print("self_bleu tri : " + str(self_bleu_tri))
    print("self_bleu four : " + str(self_bleu_four))
    print("self_bleu fif : " + str(self_bleu_fif))
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-6)
num_epochs = 5
num_training_steps = num_epochs * len(train_dataset)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps

)

progress_bar = tqdm(range(num_training_steps))


if CONTINUOUSLY_TRAIN:
    checkpoint= torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def trainer():
    model.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(train_dataset,0):
        # get the inputs; data is a list of [inputs, labels]
            mini_running_loss=0.0
        
            input_ids,attention_mask,num_decoder_input_ids,decoder_attention_masks= (data['input_ids'],data['input_attention'],data['decoder_input_ids'],data['decoder_attention_mask'])
            
        
            count=0
        
            prev_predictions=data['prompt']
        
            input_ids=input_ids.to('cuda:1')
            attention_mask=attention_mask.to('cuda:1')
            

            memory = torch.zeros_like(torch.empty(1,1024,config.d_model)).to('cuda:1') # first memory.
            #cumul_prev_predictions = torch.zeros_like(torch.empty(1,1)).to('cuda:1')
            cumul_prev_predictions=[]
        #print(prev_predictions)
            #torch.cuda.empty_cache() # manually freeing gpu memory.
            for d in num_decoder_input_ids:

                #input()
                prev_predictions=prev_predictions.to('cuda:1')
                decoder_attention_mask=decoder_attention_masks[count]
                
                word=""
                if count==0:
                    word=str(count+1) + "st"
                elif count==1:
                    word=str(count+1) + "nd"
                elif count==2:
                    word=str(count+1) + "rd"
                else:
                    word=str(count+1) + "th"
            
                prompt="MAKE A " + word + " PART OF THE ENTIRE ARTICLE. The plot : "
                
                prompt=tokenizer(prompt,return_tensors="pt")
                prompt_attention=prompt.attention_mask.to('cuda:1')
                prompt_ids=prompt.input_ids.to('cuda:1')

            #print("before concat, prompt ids shape :")
            #print(prompt_ids.shape)
                #print(decoder_attention_masks.shape) 
                dd=torch.unsqueeze(d[:-1],dim=0).to('cuda:1')
                decoder_attention_mask=torch.unsqueeze(decoder_attention_masks[count][:-1],dim=0).to('cuda:1')
            # input_ids 맨 앞에 이전 preceding context를 합친다.
                label=torch.unsqueeze(d[1:],dim=0).to('cuda:1')
            
                
                # zero the parameter gradients
                optimizer.zero_grad()

        # forward + backward + optimize
                if USE_CUMULATIVE and count>0:
                    length=len(cumul_prev_predictions)
                    #print("one step." + str(length))
                    for j in range(1,CUMUL_NUM if length>CUMUL_NUM else length):
                        #print(prev_predictions.shape)
                        if prompt_ids.shape[1]+input_ids.shape[1]+(cumul_prev_predictions[j].shape[1])+prev_predictions.shape[1]>1000:
                            #print("break")
                            #print(cumul_prev_predictions[j].shape)
                            break
                        prev_predictions=torch.cat((prev_predictions,cumul_prev_predictions[j]),1)       
                count+=1

                outputs,memory = model(memory=memory.detach(),input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = dd,decoder_attention_mask=decoder_attention_mask,labels=label,output_hidden_states=True,prev_predictions=prev_predictions,prompt_ids=prompt_ids,prompt_attention=prompt_attention) # 중요! memory.detach()를 하지 않으면 매번 memory cell에 대한 gradient는 계속 이어져나가 계산되기 때문에, 두번 그래디언트 업데이트 했다고 오류 뜬다.
                
                loss = outputs.loss
                loss.backward()
                
                prev_predictions =  torch.argmax(outputs.logits, dim=-1)
                cumul_prev_predictions.insert(0,prev_predictions)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # print statistics
                mini_running_loss += loss.item()
        
            running_loss +=mini_running_loss / count
            progress_bar.update(1)
            if i % 10 == 9:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10 :.8f}, torch saved.')
                
                writer.add_scalar("Loss/train", running_loss/10, epoch * len(train_dataset) + i)
                
                running_loss=0.0
                torch.save({'epoch':num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },PATH)

            if i % 500 == 499:
                do_eval(epoch * len(train_dataset)+i)
                writer.flush()

    print('Finished Training')

    writer.flush()
    torch.save({'epoch':num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },PATH)


trainer()
writer.close()
