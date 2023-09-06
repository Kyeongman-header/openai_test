import torch
import pickle
from tqdm import tqdm, trange
from dataset_consts import *
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig,LongformerModel
#import pytorch_metric_learning.losses as loss_fn
import sys

print("gpu : ")
print(torch.cuda.is_available())
torch.set_printoptions(linewidth=200)

save_dir=sys.argv[1] #all.tar

gpu=sys.argv[2] # cuda:0 or cpu
debug=int(sys.argv[3]) # 1 or 0

if debug==1:
    debug=True
else:
    debug=False

createFolder('longformer')
PATH = './longformer/'+save_dir


tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
with open("coherence_completeness/valid_"+"logical-2"+".pickle","rb") as fi:
    valid_dataset = pickle.load(fi)
valid_dataset= torch.utils.data.DataLoader(valid_dataset,shuffle=True)

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
        loss=0
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
     
checkpoint= torch.load(PATH)
mylongformer.load_state_dict(checkpoint['model_state_dict'],strict=False)

def eval(steps,input_text):
    valid_loss=0.0
    # acc=0
    pp=0
    nn=0
    len_pp=0
    len_nn=0
    mylongformer.eval()
    if input_text is None:
        input_text=input()
        inputs=tokenizer(input_text,max_length=4096,padding="max_length",
                truncation=True,return_tensors="pt")
    
        input_ids=inputs['input_ids']
        attention_mask=inputs['attention_mask']
        print(input_ids[0])
        print(attention_mask[0])
    else:
        input_ids=input_text['input_ids']
        attention_mask=input_text['attention_mask']
        input_text=tokenizer.batch_decode(input_ids,skip_special_tokens=True)
        inputs=tokenizer(input_text,max_length=4096,padding="max_length",
                truncation=True,return_tensors="pt")

        input_ids=inputs['input_ids']
        attention_mask=inputs['attention_mask']
        print(input_ids[0])
        print(attention_mask[0])
    if torch.cuda.is_available():
        input_ids=input_ids.to(gpu)
        attention_mask=attention_mask.to(gpu)
        global_attention_mask=torch.zeros_like(attention_mask).to(gpu)
        global_attention_mask[:,0]=1
    with torch.no_grad():
        probs,_=mylongformer(input_ids=input_ids,attention_mask=attention_mask,global_attention_mask=global_attention_mask,labels=None)
    #valid_loss += loss.item()
    if debug:

        print("probs")
        print(probs)
        #print("loss")
        #print(loss)

for i,(input_ids,attention_mask,global_attention_mask,labels) in enumerate(tqdm(valid_dataset)):
    #print(tokenizer.batch_decode(input_ids,skip_special_tokens=True))
    #print(labels)
    #eval(0,{'input_ids':input_ids,'attention_mask':attention_mask})
    
    eval(0,None)
