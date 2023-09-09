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
        input_text=tokenizer(input_text,max_length=4096,padding="max_length",
                truncation=True,return_tensors="pt")
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
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained("gpt2")
    _tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_token_id=_tokenizer.pad_token_id
    _text='''The day I turned 25 I was not surprised when I had learned the terrible truth about humanity. It was right there, out in the open, all along. All one had to do was... Accept it. It's not as though it was on purpose. It was no one's design. It just *was. *   I awoke that morning the way I awoke every morning. A bit groggy, underneath a pile of covers, the green glow of my alarm clock across the ceiling. It was still dark, and work would begin in a few hours. I would have to continue serving like a mindless drone, * '' Oh would you like coffee or tea this morning, sir? `` * How... completely meaningless. Yes I know, how does one WORK on their birthday!? Well let me tell you, friend.'''
    _input=_tokenizer(_text,max_length=250,padding="max_length",
            truncation=True,return_tensors="pt")

    print(_input)
    print(_tokenizer.batch_decode(_input['input_ids'],skip_special_tokens=True))
    eval(0,_tokenizer.batch_decode(_input['input_ids'],skip_special_tokens=True))
    _tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    _tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    _pad_token_id=tokenizer.pad_token_id
    #_text='''And if nothing horrible came to pass, I’d start to think (which is probably healthy) that what I’d been doing for years had been for naught—but in a positive way. I could leave it all behind if I was able to convince myself that what I’d applied so much meaning to actually meant nothing—that I could more than make it in the world without relying on something I was trying to convince myself was complete nonsense. At the end of the day, it all worked out. Again, the day passed without much to report. My girlfriend and I happened to exchange Christmas gifts that night, and she’d found me a new charm—this one a laptop, which was more fitting to the way I work. I’m doing my best to pretend I don’t believe it’ll bring me good fortune, but you never know what’s going to work. Though I do take my necklace off when I shower now. You never really know, do you?'''
    _input=_tokenizer(_text,max_length=250,padding="max_length",
            truncation=True,return_tensors="pt")

    print(_input)
    print(_tokenizer.batch_decode(_input['input_ids'],skip_special_tokens=True))
    eval(0,_tokenizer.batch_decode(_input['input_ids'],skip_special_tokens=True))
