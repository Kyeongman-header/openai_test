import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import AutoConfig
from dataset_consts import *

config = AutoConfig.from_pretrained('facebook/bart-large-cnn',gradient_checkpointing=True)
model =  AutoModelForSeq2SeqLM.from_config(config).to('cuda:0') # not pretrained.

from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer


TRAIN_RANGE=25000

train_total_target=last_target[:90]
train_total_source=total_target[:90]
train_total_prompt=total_source[:90]
val_total_target=last_target[90:100]
val_total_source=total_target[90:100]
val_total_prompt=total_source[90:100]

train_dataset=return_dataset_2(train_total_target,train_total_source,train_total_prompt)
valid_dataset=return_dataset_2(val_total_target,val_total_source,val_total_prompt)


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataset)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
progress_bar = tqdm(range(num_training_steps))


model.train()
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataset,0):
        # get the inputs; data is a list of [inputs, labels]
        

        input_ids,attention_mask,num_decoder_input_ids = (data['input_ids'],data['input_attention'],data['decoder_input_ids'])
        sections=["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth"]
        count=0
        prev_predictions=data['prompt']
        prev_predictions = torch.mean(prev_predictions,dim=-2)
        for d in num_decoder_input_ids:
            decoder_input_ids=d
            prompt="MAKE A " + sections[count] + " PART OF THE ENTIRE ARTICLE."
            count+=1
            prompt=tokenizer(prompt,return_tensors="pt")
            input_ids=torch.cat((prompt,input_ids),0)

            if prev_predictions: # input_ids 맨 앞에 이전 preceding context를 합친다.
                
                input_ids=torch.cat((prev_predictions,input_ids),0)

        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            outputs = model({'input_ids':input_ids,'attention_mask' : attention_mask,'decoder_input_ids' : decoder_input_ids,'labels':decoder_input_ids})
            loss = outputs.loss
            loss.backward()
            
            prev_predictions = torch.softmax(outputs.logits, dim=-1) # (,1024, 50264)를 (,1024,50264)로 바꾼다.
            prev_predictions = torch.mean(prev_predictions,dim=-2) # (,50264로 바꾼다.) 이게 맞나?

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # print statistics
            running_loss += loss.item()
        
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


print('Finished Training')


import evaluate

metric = evaluate.load("rouge")
model.eval()
for data in valid_dataset:
    input_ids,attention_mask,decoder_input_ids = (data['input_ids'],data['input_attention'],data['decoder_input_ids'])

    with torch.no_grad():
        outputs = model({'input_ids':input_ids,'attention_mask' : attention_mask,'decoder_input_ids' : decoder_input_ids,'labels':decoder_input_ids})

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=data["decoder_input_ids"])

metric.compute()