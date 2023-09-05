import pickle
import torch
from tqdm import tqdm, trange
from dataset_consts_bart import *
import random
from torch.utils.tensorboard import SummaryWriter
import evaluate
_bleu=evaluate.load("bleu")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error Creating directory. ' + directory)
createFolder('second_level')

writer = SummaryWriter("./runs/"+"real")

def do_eval(steps,dataset,NumPar,eval_num,eval_first):
    
    
    
    index=0

    N=100
    # 이건 문단 내부적으로  얼마나 반복성이 심한지 보는 지표이다.
    in_self_bleu_one=0
    in_self_bleu_bi=0
    in_self_bleu_tri=0
    in_self_bleu_four=0
    in_self_bleu_fif=0
    

    
    whole_num=0
    whole_predictions=[]
    whole_predictions_len=0
    whole_labels=[]
    whole_labels_len=0

    for i in trange(0, eval_num, batch_size):
        if i+batch_size>eval_num or i+batch_size>len(dataset):
            # batch size에 안 맞는 마지막 set은 , 그냥 버린다
            # batch size는 커봐야 4 정도니까 이정도는 괜찮다.
            if i<=1: # 데이터셋이 단 한개 이하로 있는 경우
                # self bleu 등을 구할 수 없기 때문에 그냥 넘긴다. 
                return
            break
    # get the inputs; data is a list of [inputs, labels]]
        batch_data=dataset[i:i+batch_size]
        first=True
        batch_num_decoder_input_ids=[]
        batch_decoder_attention_masks=[]
        one_prediction=[]
        for i in range(batch_size):
            one_prediction.append([])
        one_label=[]
        for j in range(batch_size):
            one_label.append([])
        for data in batch_data:
            input_ids,attention_mask,num_decoder_input_ids,decoder_attention_masks,prompt= (data['input_ids'],data['input_attention'],data['decoder_input_ids'],data['decoder_attention_mask'],data['prompt'])
            batch_num_decoder_input_ids.append(num_decoder_input_ids)
            batch_decoder_attention_masks.append(decoder_attention_masks)
            if first:
                batch_input_ids=input_ids
                batch_attention_mask=attention_mask
                batch_prev_predictions=prompt
                first=False
            else:
                batch_input_ids=torch.cat((batch_input_ids,input_ids),dim=0)
                batch_attention_mask=torch.cat((batch_attention_mask,attention_mask),dim=0)
                
                batch_prev_predictions=torch.cat((batch_prev_predictions,prompt),dim=0)
        batch_num_decoder_input_ids=torch.stack(batch_num_decoder_input_ids,dim=1)
        batch_decoder_attention_masks=torch.stack(batch_decoder_attention_masks,dim=1)
        # print("batch dataset shapes")
        # print(batch_input_ids.shape) #(b, 200)
        # print(batch_attention_mask.shape) 
        # print(batch_num_decoder_input_ids.shape) # (N,b,250)
        # print(batch_decoder_attention_masks.shape) # (N, b, 250)
        # print(batch_prev_predictions.shape) #(b,150)
        # print("batch dataset shape ends")
        batch_input_ids=batch_input_ids
        batch_attention_mask=batch_attention_mask
        batch_num_decoder_input_ids=batch_num_decoder_input_ids
        batch_decoder_attention_masks=batch_decoder_attention_masks
        batch_prev_predictions=batch_prev_predictions
        
        count=0
    
        
        
    #print(prev_predictions)
        #torch.cuda.empty_cache() # manually freeing gpu memory.
        for count in range(NumPar):

            _batch_decoder_input_ids=batch_num_decoder_input_ids[count] #(b,250)
            _batch_decoder_attention_masks=batch_decoder_attention_masks[count] #(b,250)

            
            # dd=torch.unsqueeze(decoder_input_id[:-1],dim=0)
            # decoder_attention_mask=torch.unsqueeze(decoder_attention_masks[count][:-1],dim=0)
        # input_ids 맨 앞에 이전 preceding context를 합친다.
            # label=torch.unsqueeze(d[1:],dim=0)
            # _batch_labels=_batch_decoder_input_ids[:,1:] #(b,249)
            _batch_labels=_batch_decoder_input_ids #(b,250)
            _batch_decoder_input_ids=_batch_decoder_input_ids[:,:-1] #(b,249)
            _batch_decoder_attention_masks=_batch_decoder_attention_masks[:,:-1] #(b,249)

            
            labels = tokenizer.batch_decode(_batch_labels,skip_special_tokens=True)
            

            _predictions_len=0
            _labels_len=0

            # for u,pred in enumerate(predictions):
                
            #     _pred= tokenizer(pred,return_tensors="pt")['input_ids']
            #     _predictions_len+=len(_pred[0])
            #     one_prediction[u].append(pred)
            
            for u,lab in enumerate(labels):
                _lab= tokenizer(lab,return_tensors="pt")['input_ids']
                _labels_len+=len(_lab[0])
                one_label[u].append(lab)

            
        # print("whole predict")
        # print(one_prediction)
        # print("whole label")
        # print(one_label)

        for u,_one_label in enumerate(one_label):
            _in_self_bleu_one=0
            _in_self_bleu_bi=0
            _in_self_bleu_tri=0
            _in_self_bleu_four=0
            _in_self_bleu_fif=0

            whole_one_label=' '.join(one_label[u])


            if len(_one_label)>1:
                for j in range(len(_one_label)): 
                    except_one_label=_one_label[0:j]+_one_label[j+1:]
                    # print("except one")
                    # print(except_one_prediction)
                    # print("one")
                    # print(_one_prediction[j])
                    # 학습이 제대로 안되서 generate 길이가 0이면, 이게 제대로 작동 안한다.
            #self_bleu=BLEU(except_whole_predictions,weights).get_score([whole_predictions[j]])
                    self_bleu=_bleu.compute(predictions=[_one_label[j]],references=[except_one_label],max_order=5)
                    _in_self_bleu_one+=self_bleu['precisions'][0]
                    _in_self_bleu_bi+=self_bleu['precisions'][1]
                    _in_self_bleu_tri+=self_bleu['precisions'][2]
                    _in_self_bleu_four+=self_bleu['precisions'][3]
                    _in_self_bleu_fif+=self_bleu['precisions'][4]

            in_self_bleu_one+=_in_self_bleu_one/len(_one_label)
            in_self_bleu_bi+=_in_self_bleu_bi/len(_one_label)
            in_self_bleu_tri+=_in_self_bleu_tri/len(_one_label)
            in_self_bleu_four+=_in_self_bleu_four/len(_one_label)
            in_self_bleu_fif+=_in_self_bleu_fif/len(_one_label)
        
        
            if len(whole_one_label)==0:
                print("something wrong. label is not exist.")
                print("error set : ")
                
                print("label : ")
                print(whole_one_label)
                continue
            whole_labels.append(whole_one_label)
            whole_labels_len+=_labels_len
            whole_num+=1
        

   
    

    in_self_bleu_one=in_self_bleu_one/whole_num
    in_self_bleu_bi=in_self_bleu_bi/whole_num
    in_self_bleu_tri=in_self_bleu_tri/whole_num
    in_self_bleu_four=in_self_bleu_four/whole_num
    in_self_bleu_fif=in_self_bleu_fif/whole_num
    print("in_self_bleu_one : " + str(in_self_bleu_one))
    print("in_self_bleu_bi : " + str(in_self_bleu_bi))
    print("in_self_bleu_tri : " + str(in_self_bleu_tri))
    print("in_self_bleu_four : " + str(in_self_bleu_four))
    print("in_self_bleu_fif : " + str(in_self_bleu_fif))
    writer.add_scalar("real_in self bleu one/eval", in_self_bleu_one, steps)
    writer.add_scalar("real_in self bleu bi/eval", in_self_bleu_bi, steps)
    writer.add_scalar("real_in self bleu tri/eval", in_self_bleu_tri, steps)
    writer.add_scalar("real_in self bleu four/eval", in_self_bleu_four, steps)
    writer.add_scalar("real_in self bleu fif/eval", in_self_bleu_fif, steps)

for i in range(1,20): # 최대 100개 문단까지 있다.
        
        
        with open("pickle_data/"+"bart_test_"+"wp_rake"+"/level_2_" + str(i) + ".pickle","rb") as fi:
            test_dataset = pickle.load(fi)
        with open("pickle_data/"+"bart_valid_"+"reedsy_rake"+"/level_2_" + str(i) + ".pickle","rb") as fi:# reedsy rake는 test dataset이 없다
            test_dataset += pickle.load(fi)
        with open("pickle_data/"+"bart_test_"+"booksum_rake"+"/level_2_" + str(i) + ".pickle","rb") as fi:
            test_dataset += pickle.load(fi)
        
        if len(test_dataset)==0:
            continue
        
        print("the test set for " + str(i) + " Num Paragramphs.")

        do_eval(steps=i,dataset=test_dataset,NumPar=i,eval_num=80,eval_first=eval_first)
        eval_first=False