import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import AutoConfig
from dataset_consts import *

config = AutoConfig.from_pretrained('t5-base',gradient_checkpointing=True)
t5 =  AutoModelForSeq2SeqLM.from_config(config).to('cuda:0') # not pretrained.

from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer


TRAIN_RANGE=25000

PATH = './second_level/'+'all.tar'

CONTINUOUSLY_TRAIN=True

train_total_target=last_target[:5]
train_total_source=total_target[:5]
train_total_prompt=total_source[:5]
val_total_target=last_target[5:8]
val_total_source=total_target[5:8]
val_total_prompt=total_source[5:8]

train_dataset=return_dataset_2(train_total_target,train_total_source,train_total_prompt)
valid_dataset=return_dataset_2(val_total_target,val_total_source,val_total_prompt)


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler



print(config.vocab_size)
print(config.d_model)


class Network(nn.Module): 
   def __init__(self, vocab_size, d_model,t5): 
       super(Network, self).__init__() 
        
       self.shared = nn.Embedding(config.vocab_size, config.d_model) 
       self.t5 = t5

   def forward(self, input_ids,attention_mask,decoder_input_ids,labels,output_hidden_states,prev_predictions,prompt_ids,prompt_attention):

       
       prev_predictions = self.shared(prev_predictions)
       
       #print("embedded prev prediction : ")
       #print(prev_predictions.shape)
       
       prev_predictions = torch.mean(prev_predictions,dim=-2,keepdim=True)
       
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
       
       inputs_embeds=torch.cat((prev_predictions,inputs_embeds),1)
       attention_mask=torch.cat((torch.LongTensor([[1]]).to('cuda:0'),attention_mask),1)

       #print("prev concat input embeds shape : ")
       #print(inputs_embeds.shape)
       #print("prev concat attention mask shape : ")
       #print(attention_mask.shape)
       outputs = self.t5(input_ids = None,inputs_embeds=inputs_embeds,attention_mask = attention_mask,decoder_input_ids = decoder_input_ids,labels=decoder_input_ids,output_hidden_states=True)
       return outputs


model = Network(config.vocab_size, config.d_model,t5).to('cuda:0')


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
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


model.train()
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(train_dataset,0):
        # get the inputs; data is a list of [inputs, labels]
        mini_running_loss=0.0
        
        input_ids,attention_mask,num_decoder_input_ids = (data['input_ids'],data['input_attention'],data['decoder_input_ids'])
        sections=["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth"]
        
        count=0
        
        prev_predictions=data['prompt']
        
        input_ids=input_ids.to('cuda:0')
        attention_mask=attention_mask.to('cuda:0')
        
        #print(prev_predictions)        
        for d in num_decoder_input_ids:

            #input()
            prev_predictions=prev_predictions.to('cuda:0')
            
            prompt="MAKE A " + sections[count] + " PART OF THE ENTIRE ARTICLE."
            count+=1
            prompt=tokenizer(prompt,return_tensors="pt")
            prompt_attention=prompt.attention_mask.to('cuda:0')
            prompt_ids=prompt.input_ids.to('cuda:0')

            #print("before concat, prompt ids shape :")
            #print(prompt_ids.shape)
            
            d=torch.unsqueeze(d,dim=0).to('cuda:0')

            # input_ids 맨 앞에 이전 preceding context를 합친다.
                
            

        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            outputs = model(input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = d,labels=d,output_hidden_states=True,prev_predictions=prev_predictions,prompt_ids=prompt_ids,prompt_attention=prompt_attention)

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
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000 :.8f}')
            running_loss = 0.0


print('Finished Training')



torch.save({'epoch':num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },PATH)


# -----------train ends, eval starts.

import evaluate

metric = evaluate.load("rouge")
model.eval()
for data in valid_dataset:
    input_ids,attention_mask,num_decoder_input_ids = (data['input_ids'],data['input_attention'],data['decoder_input_ids'])
    sections=["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth"]

    count=0

    prev_predictions=data['prompt']

    input_ids=input_ids.to('cuda:0')
    attention_mask=attention_mask.to('cuda:0')

        #print(prev_predictions)
    for d in num_decoder_input_ids:

            #input()
        prev_predictions=prev_predictions.to('cuda:0')

        prompt="MAKE A " + sections[count] + " PART OF THE ENTIRE ARTICLE."
        count+=1
        prompt=tokenizer(prompt,return_tensors="pt")
        prompt_attention=prompt.attention_mask.to('cuda:0')
        prompt_ids=prompt.input_ids.to('cuda:0')

            #print("before concat, prompt ids shape :")
            #print(prompt_ids.shape)

        ex_d=torch.unsqueeze(d,dim=0).to('cuda:0')

            # input_ids 맨 앞에 이전 preceding context를 합친다.
        with torch.no_grad():
            outputs = model(input_ids = input_ids,attention_mask = attention_mask,decoder_input_ids = ex_d,labels=ex_d,output_hidden_states=True,
                    prev_predictions=prev_predictions,prompt_ids=prompt_ids,prompt_attention=prompt_attention)

        prev_predictions =  torch.argmax(outputs.logits, dim=-1)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predictions = tokenizer.decode(predictions,skip_special_tokens=True)
        ex_d = tokenizer.decode(ex_d,skip_special_tokens=True)
        
        metric.add_batch(predictions=predictions, references=ex_d)

print(metric.compute())
