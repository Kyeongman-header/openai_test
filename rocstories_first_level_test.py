import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import AutoConfig
from dataset_consts import *
from torch.utils.data import DataLoader

pretrained_location="./checkpoint-125000"

model=AutoModelForSeq2SeqLM.from_pretrained(pretrained_location)

with open("pickle_data/"+"ROCStories_train"+"/level_1.pickle","rb") as fi:
    train_dataset = pickle.load(fi)
with open("pickle_data/"+"ROCStories_valid"+"/level_1.pickle","rb") as fi:
    valid_dataset = pickle.load(fi)
    
import evaluate

metric = evaluate.load("rouge")


f = open('first_level_train_results.csv','w', newline='')
wr = csv.writer(f)
wr.writerow(["index","source","real text","generated_results"])
index=0

model.eval()
for data in tqdm(DataLoader(torch.utils.data.Subset(train_dataset,list(range(0,1000))), batch_size=4)):
    
    input_ids,attention_mask,decoder_input_ids = (data['input_ids'],data['attention_mask'],data['decoder_input_ids'])
    print(input_ids.shape)
    print(attention_mask.shape)
    for i in range((input_ids.shape[0])):
        
        print(decoder_input_ids[i])
        #print(input_ids[i])

        predicted_abstract_ids = model.generate(torch.unsqueeze(input_ids[i],dim=0), attention_mask=torch.unsqueeze(attention_mask[i],dim=0),num_beams=1) #max_length=512,encoder_no_repeat_ngram_size=3,no_repeat_ngram_size=3,repetition_penalty=3.5,num_beams=4,early_stopping=True,)
        #print("generation ends")
        """for j in range(1,len(decoder_input_ids[i])):
            print(decoder_input_ids[i][:j])
            print(tokenizer.decode(decoder_input_ids[i][:j]))
            print("model through")
            predicted = model(torch.unsqueeze(input_ids[i],dim=0), attention_mask=torch.unsqueeze(attention_mask[i],dim=0), decoder_input_ids=torch.unsqueeze(decoder_input_ids[i][:j],dim=0),)
            predicted_abstract_ids_2=torch.argmax(predicted.logits, dim=-1)
            predictions_2= tokenizer.batch_decode(predicted_abstract_ids_2,skip_special_tokens=True)
            
            print(predictions_2)
            input()
        #print(tokenizer.bos_token_id)
        """
        #predicted = model(torch.unsqueeze(input_ids[i],dim=0), attention_mask=torch.unsqueeze(attention_mask[i],dim=0), decoder_input_ids=torch.tensor([[tokenizer.bos_token_id,2515]]),)
        print(predicted_abstract_ids)
    
        #predicted_abstract_ids_2=torch.argmax(predicted.logits, dim=-1)
        predictions=tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
        #predictions_2= tokenizer.batch_decode(predicted_abstract_ids_2,skip_special_tokens=True)
        golden_labels=tokenizer.batch_decode(torch.unsqueeze(decoder_input_ids[i],dim=0),skip_speical_tokens=True)        
        print(index)
        index+=1
        print("prompt : ")
        print(tokenizer.decode(input_ids[i],skip_special_tokens=True))
        print()
        print("golden labels : ")
        print(golden_labels)
        print("predictions : ")
        print(predictions)
        #print("predictions_2 : ")
        #print(predictions_2)

        metric.add_batch(predictions=predictions, references=golden_labels)

print(metric.compute())

"""
    for i in range((input_ids.shape[0])):
        wr.writerow([str(index),tokenizer.decode(input_ids[i],skip_special_tokens=True),golden_labels[i],predictions[i]])
        print(index)
        #print("loss")
        #print(predicted.loss)
        print("prompt : ")
        print(tokenizer.decode(input_ids[i],skip_special_tokens=True))
        print()
        print("golden labels : ")
        print(golden_labels[i])
        print("predictions : ")
        print(predictions[i])
        index+=1
        metric.add_batch(predictions=predictions, references=golden_labels)

print(metric.compute())"""
