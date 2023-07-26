
import torch
import pickle
from tqdm import tqdm, trange
from dataset_consts import *
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig,LongformerModel
import sys

print("gpu : ")
print(torch.cuda.is_available())

tdataset_name=sys.argv[1] # 예제 : coherence-1
vdataset_name=sys.argv[2] # 예제 : coherence
num_epochs=int(sys.argv[3]) # 예제 : 1
num_batch =int(sys.argv[4]) # 예제 : 4
dtest=int(sys.argv[5]) # 1 or 0
do_test=False
if dtest==1:
    do_test=True
num_test=int(sys.argv[6]) # 4
conti=int(sys.argv[7]) # 1 or 0
save_dir=sys.argv[8] #all.tar
log_dir=sys.argv[9] # coh1
gpu=sys.argv[10] # cuda:0 or cpu
debug=int(sys.argv[11]) # 1 or 0

if debug==1:
    debug=True
else:
    debug=False

print("train dataset : " + tdataset_name)
print("valid dataset : " + vdataset_name)
print("epochs : " + str(num_epochs))
print("batch(only for train) : " + str(num_batch))
print("do test ? ")
print(dtest)
print("num test : " + str(num_test))
print("continuous train ?" + str(conti))
print("save dir : " + save_dir)
print("log dir : " + log_dir)
print("gpu or cpu : " + gpu)
print("debug mode : " + str(debug))




if tdataset_name=="coherence-whole" or vdataset_name=="coherence-whole":
    with open("coherence_completeness/train_"+"coherence-new-1"+".pickle","rb") as fi:
            train_dataset = pickle.load(fi)
    with open("coherence_completeness/valid_"+"coherence-new-1"+".pickle","rb") as fi:
            valid_dataset = pickle.load(fi)
    with open("coherence_completeness/train_"+"coherence-new-2"+".pickle","rb") as fi:
            train_dataset = train_dataset + pickle.load(fi)
    with open("coherence_completeness/valid_"+"coherence-new-2"+".pickle","rb") as fi:
            valid_dataset = valid_dataset + pickle.load(fi)
    with open("coherence_completeness/train_"+"coherence-new-3"+".pickle","rb") as fi:
            train_dataset += train_dataset + pickle.load(fi)
    with open("coherence_completeness/valid_"+"coherence-new-3"+".pickle","rb") as fi:
            valid_dataset += valid_dataset + pickle.load(fi)
elif tdataset_name=="completeness-whole" or vdataset_name=="completeness-whole":
    with open("coherence_completeness/train_"+"completeness-new-1"+".pickle","rb") as fi:
            train_dataset = pickle.load(fi)
    with open("coherence_completeness/valid_"+"completeness-new-1"+".pickle","rb") as fi:
            valid_dataset = pickle.load(fi)
    with open("coherence_completeness/train_"+"completeness-new-2"+".pickle","rb") as fi:
            train_dataset = train_dataset + pickle.load(fi)
    with open("coherence_completeness/valid_"+"completeness-new-2"+".pickle","rb") as fi:
            valid_dataset = valid_dataset + pickle.load(fi)
    with open("coherence_completeness/train_"+"completeness-new-3"+".pickle","rb") as fi:
            train_dataset += train_dataset + pickle.load(fi)
    with open("coherence_completeness/valid_"+"completeness-new-3"+".pickle","rb") as fi:
            valid_dataset += valid_dataset + pickle.load(fi)
else: 
    with open("coherence_completeness/train_"+tdataset_name+".pickle","rb") as fi:
            train_dataset = pickle.load(fi)
    with open("coherence_completeness/valid_"+vdataset_name+".pickle","rb") as fi:
            valid_dataset = pickle.load(fi)

train_dataset= torch.utils.data.DataLoader(train_dataset,
                                   batch_size=num_batch,shuffle=True)
valid_dataset= torch.utils.data.DataLoader(valid_dataset,shuffle=True)

CONTINUOUSLY_TRAIN=False
if conti==1:
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

def eval(steps):
    mylongformer.eval()
    valid_loss=0.0
    acc=0
    for i,(input_ids,attention_mask,global_attention_mask,labels) in enumerate(tqdm(valid_dataset)):
        # print(input_ids.shape)
        if torch.cuda.is_available():
             input_ids=input_ids.to(gpu)
             attention_mask=attention_mask.to(gpu)
             global_attention_mask=global_attention_mask.to(gpu)
             labels=labels.to(gpu)

        probs,loss=mylongformer(input_ids=input_ids,attention_mask=attention_mask,global_attention_mask=global_attention_mask,labels=labels)
        valid_loss += loss.item()
        if debug:
            print("probs")
            print(probs)
            print("loss")
            print(loss)

        for (j,p) in enumerate(probs,0):
            if p >=0.5 and labels[j]==1:
                acc+=1
            elif p < 0.5 and labels[j]==0:
                acc+=1
        
        if debug:
            print("acc")
            print(acc)
            input()

        del input_ids
        del attention_mask
        del global_attention_mask
        del labels
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
    
    valid_loss=(valid_loss/len(valid_dataset))
    print("valid loss : ")
    print(valid_loss)
    print("valid acc : ")
    print(acc/len(valid_dataset))
    writer.add_scalar("loss/valid",valid_loss, steps)
    writer.add_scalar("acc/valid",acc/len(valid_dataset),steps)

optimizer = optim.AdamW(mylongformer.parameters(), lr=1e-5)
num_training_steps = num_epochs * len(train_dataset)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=50000, num_training_steps=num_training_steps
)

if CONTINUOUSLY_TRAIN:
    checkpoint= torch.load(PATH)
    mylongformer.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for epoch in range(num_epochs):
    running_loss = 0.0
    mylongformer.train()
    
    loss_report=3000

    model_save=3000
    eval_report=2
    loss_steps=1
    acc=0
    #eval_steps=1
    #eval(0)
    
    for i,(input_ids,attention_mask,global_attention_mask,labels) in enumerate(tqdm(train_dataset)):
        # print(input_ids.shape)
        if torch.cuda.is_available():
             input_ids=input_ids.to(gpu)
             attention_mask=attention_mask.to(gpu)
             global_attention_mask=global_attention_mask.to(gpu)
             labels=labels.to(gpu)

        prob,loss=mylongformer(input_ids=input_ids,attention_mask=attention_mask,global_attention_mask=global_attention_mask,labels=labels)
        if debug:
            print("prob")
            print(prob)
            print("loss")
            print(loss)
        for (j,p) in enumerate(prob,0):
            if p >=0.5 and labels[j]==1:
                acc+=1
            elif p < 0.5 and labels[j]==0:
                acc+=1
        
        
        if debug:
            print(acc)
            input()
        
        # print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()


        if i%loss_report==(loss_report-1):
            print(running_loss/(loss_report*num_batch))
            print(acc/(loss_report*num_batch))
            writer.add_scalar("loss/train",running_loss/(loss_report*num_batch),loss_steps)
            writer.add_scalar("acc/train",acc/(loss_report*num_batch),loss_steps)
            running_loss=0
            acc=0
            loss_steps+=1

            # input()
        
        
        if i%model_save==model_save-1:
             torch.save({'epoch':num_epochs,
            'model_state_dict': mylongformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },PATH)
        
        del input_ids
        del attention_mask
        del global_attention_mask
        del labels
             
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
        """
        if i%eval_report==eval_report-1:
            eval(eval_steps)
            eval_steps+=1
        """
    if do_test:
        eval(num_test)


print('Finished Training')
