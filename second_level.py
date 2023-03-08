import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import AutoConfig
from dataset_consts import *


from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer


TRAIN_RANGE=25000
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error Creating directory. ' + directory)

createFolder('second_level')
PATH = './second_level/'+'all.tar'

CONTINUOUSLY_TRAIN=False

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
       
       
       prev_predictions=torch.cat((torch.LongTensor([[tokenizer.pad_token_id]*(1024-prev_predictions.shape[1])]).to('cuda:1'),prev_predictions),1)
       prev_predictions = self.shared(prev_predictions)
       #print(prev_predictions.shape)

       
       memory=self.grucell(torch.squeeze(prev_predictions),torch.squeeze(memory)).unsqueeze(dim=0)
       
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
       inputs_embeds=self.shared(input_ids)
       attention_mask=torch.cat((prompt_attention,attention_mask),1)
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
    
   def generate(self, memory,input_ids,attention_mask,decoder_input_ids,labels,output_hidden_states,prev_predictions,prompt_ids,prompt_attention):
       
       prev_predictions=torch.cat((torch.LongTensor([[tokenizer.pad_token_id]*(1024-prev_predictions.shape[1])]).to('cuda:1'),prev_predictions),1)
       prev_predictions = self.shared(prev_predictions)
       #print(prev_predictions.shape)


       memory=self.grucell(torch.squeeze(prev_predictions),torch.squeeze(memory)).unsqueeze(dim=0)
       #print("embedded prev prediction : ")
       #print(prev_predictions.shape)

       #prev_predictions = torch.mean(prev_predictions,dim=-2,keepdim=True)

       #print("avg embedded prev prediction : ")
       #print(prev_predictions.shape)


       #print("before concat, prompt ids shape :")
       #print(prompt_ids.shape)
       #print("before concat, input ids shape :")
       #print(input_ids.shape)

       input_ids=torch.cat((prompt_ids,input_ids),1)
       inputs_embeds=self.shared(input_ids)
       attention_mask=torch.cat((prompt_attention,attention_mask),1)

       #print("concat and embedded input ids shape :")
       #print(inputs_embeds.shape)
       #print("concat attention mask shape : ")
       #print(attention_mask.shape)

       #inputs_embeds=torch.cat((prev_predictions,inputs_embeds),1)
       #attention_mask=torch.cat((torch.LongTensor([[1]*prev_predictions.shape[1]]).to('cuda:1'),attention_mask),1)
       dummy_decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to('cuda:1')
       source= tokenizer.batch_decode(input_ids,skip_special_tokens=True)
       print("source")
       print(source)
       return self.bart.generate(max_length=512,memory=memory,inputs_embeds=inputs_embeds,attention_mask=attention_mask,decoder_input_ids=dummy_decoder_input_ids),memory

config = AutoConfig.from_pretrained('facebook/bart-base',gradient_checkpointing=True)
if CONTINUOUSLY_TRAIN:
    bart =  AutoModelForSeq2SeqLM.from_config(config).to('cuda:1') # 이후부터는 내가 finetune한 bart를 사용(밑에서 torch로 불러온다.)
else:
    bart = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base').to('cuda:1') # 최초 학습에서는 pretrained 된 bart를 사용

bart.get_input_embeddings().requires_grad = False # embedding layer는 학습을 안한다. 얘가 변동되면 prev_predictions에 대한 표현도 계속 변하기 때문.

model = Network(config.vocab_size, config.d_model,bart).to('cuda:1')


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-6)
num_epochs = 3
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
                count+=1
                prompt=tokenizer(prompt,return_tensors="pt")
                prompt_attention=prompt.attention_mask.to('cuda:1')
                prompt_ids=prompt.input_ids.to('cuda:1')

            #print("before concat, prompt ids shape :")
            #print(prompt_ids.shape)
            
                dd=torch.unsqueeze(d[:-1],dim=0).to('cuda:1')
                decoder_attention_mask=torch.unsqueeze(decoder_attention_masks[count][:-1],dim=0).to('cuda:1')
            # input_ids 맨 앞에 이전 preceding context를 합친다.
                label=torch.unsqueeze(d[1:],dim=0).to('cuda:1')
            

        # zero the parameter gradients
                optimizer.zero_grad()

        # forward + backward + optimize
                
                outputs,memory = model(memory=memory.detach(),input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = dd,decoder_attention_mask=decoder_attention_mask,labels=label,output_hidden_states=True,prev_predictions=prev_predictions,prompt_ids=prompt_ids,prompt_attention=prompt_attention) # 중요! memory.detach()를 하지 않으면 매번 memory cell에 대한 gradient는 계속 이어져나가 계산되기 때문에, 두번 그래디언트 업데이트 했다고 오류 뜬다.
                
                loss = outputs.loss
                loss.backward()
                
                prev_predictions =  torch.argmax(outputs.logits, dim=-1)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # print statistics
                mini_running_loss += loss.item()
        
            running_loss +=mini_running_loss / count
            progress_bar.update(1)
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000 :.8f}, torch saved.')
                running_loss = 0.0

                torch.save({'epoch':num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },PATH)


    print('Finished Training')


    torch.save({'epoch':num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },PATH)


trainer()

# -----------train ends, eval starts.
f = open('second_level_val_results.csv','w', newline='')
wr = csv.writer(f)
wr.writerow(["index","source","real text","generated_results"])
index=0

import evaluate

metric = evaluate.load("rouge")
model.eval()

for data in valid_dataset:
    input_ids,attention_mask,num_decoder_input_ids = (data['input_ids'],data['input_attention'],data['decoder_input_ids'])
    

    count=0

    prev_predictions=data['prompt']

    input_ids=input_ids.to('cuda:1')
    attention_mask=attention_mask.to('cuda:1')
    memory = torch.zeros_like(torch.empty(1,1024,config.d_model)).to('cuda:1') # first memory.
        #print(prev_predictions)
    for d in num_decoder_input_ids:

            #input()
        prev_predictions=prev_predictions.to('cuda:1')
        
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
        
        count+=1
        prompt=tokenizer(prompt,return_tensors="pt")
        prompt_attention=prompt.attention_mask.to('cuda:1')
        prompt_ids=prompt.input_ids.to('cuda:1')

            #print("before concat, prompt ids shape :")
            #print(prompt_ids.shape)

        ex_d=torch.unsqueeze(d,dim=0).to('cuda:1')

            # input_ids 맨 앞에 이전 preceding context를 합친다.
        #with torch.no_grad():
        #    outputs = model(input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = ex_d,labels=ex_d,output_hidden_states=True,
        #            prev_predictions=prev_predictions,prompt_ids=prompt_ids,prompt_attention=prompt_attention)

        #prev_predictions =  torch.argmax(outputs.logits, dim=-1)
        
        outputs,memory=model.generate(memory=memory.detach(),input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = ex_d,labels=ex_d,output_hidden_states=True,
                    prev_predictions=prev_predictions,prompt_ids=prompt_ids,prompt_attention=prompt_attention)
        prev_predictions = outputs # 이렇게 만들면 outputs에 id가 나오는 모양임.
        #predictions = torch.argmax(logits, dim=-1)
        
        
        predictions = tokenizer.batch_decode(outputs,skip_special_tokens=True)
        print("predictions")
        print(predictions) 
        ex_d = tokenizer.batch_decode(ex_d,skip_special_tokens=True)
        print("golden label")
        print(ex_d) 
        wr.writerow([str(index),tokenizer.batch_decode(input_ids,skip_special_tokens=True),ex_d,predictions])
        index+=1
        metric.add_batch(predictions=predictions, references=ex_d)

print(metric.compute())
