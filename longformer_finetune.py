
from transformers import AutoTokenizer, LongformerModel,AutoConfig
import torch
import pickle
from tqdm import tqdm, trange
from dataset_consts import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./runs/')

createFolder('longformer')
PATH = './longformer/'+'all.tar'

CONTINUOUSLY_TRAIN=False

print("gpu : ")
print(torch.cuda.empty_cache())

with open("train_coherence.pickle","rb") as fi:
        train_dataset = pickle.load(fi)
with open("valid_coherence.pickle","rb") as fi:
        valid_dataset = pickle.load(fi)

train_dataset= torch.utils.data.DataLoader(train_dataset,
                                   batch_size=2,
                                   shuffle=True)


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
     mylongformer=mylongformer.to('cuda:0')
if CONTINUOUSLY_TRAIN:
    checkpoint= torch.load(PATH)
    mylongformer.load_state_dict(checkpoint['model_state_dict'])
    mylongformer.load_state_dict(checkpoint['optimizer_state_dict'])

def eval(steps):
    mylongformer.eval()
    valid_loss=0.0
    for i,(input_ids,attention_mask,global_attention_mask,labels) in enumerate(tqdm(valid_dataset)):
        # print(input_ids.shape)
        if torch.cuda.is_available():
             input_ids=input_ids.to('cuda:0')
             attention_mask=attention_mask.to('cuda:0')
             global_attention_mask=global_attention_mask.to('cuda:0')
             labels=labels.to('cuda:0')

        _,loss=mylongformer(input_ids=input_ids,attention_mask=attention_mask,global_attention_mask=global_attention_mask,labels=labels)
        valid_loss += loss.item()
    
    valid_loss=(valid_loss/len(valid_dataset))
    print("valid loss : " + valid_loss)
    writer.add_scalar("loss/valid",valid_loss, steps)


num_epochs=5
optimizer = optim.AdamW(mylongformer.parameters(), lr=5e-6)
num_epochs = 5
num_training_steps = num_epochs * len(train_dataset)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps

)


for epoch in range(num_epochs):
    running_loss = 0.0
    mylongformer.train()
    loss_report=2
    model_save=2
    eval_report=2
    
    loss_steps=1
    eval_steps=1
    for i,(input_ids,attention_mask,global_attention_mask,labels) in enumerate(tqdm(train_dataset)):
        # print(input_ids.shape)
        if torch.cuda.is_available():
             input_ids=input_ids.to('cuda:0')
             attention_mask=attention_mask.to('cuda:0')
             global_attention_mask=global_attention_mask.to('cuda:0')
             labels=labels.to('cuda:0')

        prob,loss=mylongformer(input_ids=input_ids,attention_mask=attention_mask,global_attention_mask=global_attention_mask,labels=labels)
        
        # print(prob)
        # print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()


        if i%loss_report==(loss_report-1):
            print(running_loss/loss_report)
            writer.add_scalar("loss/train",running_loss/loss_report,loss_steps)
            running_loss=0
            loss_steps+=1
            # input()
        
        
        if i%model_save==model_save-1:
             torch.save({'epoch':num_epochs,
            'model_state_dict': mylongformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },PATH)
        
        if i%eval_report==eval_report-1:
             eval(eval_steps)
             eval_steps+=1

        del input_ids
        del attention_mask
        del global_attention_mask
        del labels
             
        if torch.cuda.is_available():
             torch.cuda.empty_cache()

print('Finished Training')