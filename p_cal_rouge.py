import csv
import ctypes as ct
import math
import numpy as np
from transformers import AutoTokenizer
from rouge import Rouge
import copy
rouge = Rouge()
tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large")
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
import sys
import nltk.translate.bleu_score as bleu
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()
from rake_nltk import Rake
from tqdm import tqdm, trange
testfile_name=sys.argv[1]
num_para=int(sys.argv[2])

f = open(testfile_name+'.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
num_whole_steps=sum(1 for row in rdr)
f.seek(0)
rdr = csv.reader(f)
first=True

count=0
whole_labels=[]
whole_predictions=[]
whole_labels_len=[]
whole_predictions_len=[]
one_fake=[]
one_label=[]
in_self_bleu_one=0
in_self_bleu_bi=0
in_self_bleu_tri=0
in_self_bleu_four=0
in_self_bleu_fif=0
fake_in_self_bleu_one=0
fake_in_self_bleu_bi=0
fake_in_self_bleu_tri=0
fake_in_self_bleu_four=0
fake_in_self_bleu_fif=0

whole_num=0
document_fake=[]
document_label=[]
progress_bar = tqdm(range(num_whole_steps))
for line in rdr:
    
    progress_bar.update(1)
    if first:
        first=False
        continue
    
    if line[1]!=prompts:
        document_fake.append(' '.join(one_fake))
        document_label.append(' '.join(one_label))
    prompts=line[1]
    fake=line[4]
    real=line[3]
    label_len=len(tokenizer(real,return_tensors="pt")['input_ids'][0])
    predict_len=len(tokenizer(fake,return_tensors="pt")['input_ids'][0])
    #print(label_len)
    #print(predict_len)
    whole_predictions_len.append(predict_len)
    whole_labels_len.append(label_len)
    
    whole_predictions.append(fake)
    whole_labels.append(real)

    count+=1
    one_fake.append(fake)
    one_label.append(real)
    if (len(one_fake)) % num_para==0:
        _in_self_bleu_one=0
        _in_self_bleu_bi=0
        _in_self_bleu_tri=0
        _in_self_bleu_four=0
        _in_self_bleu_fif=0

            

        for j in range(len(one_label)): 
            except_one_label=one_label[0:j]+one_label[j+1:]
            refs=[]
            for predicts in except_one_label:
                refs.append(tweet_tokenizer.tokenize(predicts))
            #print(refs)
            hyp=tweet_tokenizer.tokenize(one_label[j])
            #print(hyp)
            self_bleu=bleu.sentence_bleu(refs,hyp,weights=[(1./2.,1./2.),(1./3.,1./3.,1./3.),(1./4.,1./4.,1./4.,1./4.),(1./5.,1./5.,1./5.,1./5.,1./5.)])
            #print(self_bleu)
            _in_self_bleu_bi+=self_bleu[0]
            _in_self_bleu_tri+=self_bleu[1]
            _in_self_bleu_four+=self_bleu[2]
            _in_self_bleu_fif+=self_bleu[3]
            """
            self_bleu=_bleu.compute(predictions=[_one_label[j]],references=[except_one_label],max_order=5)
            _in_self_bleu_one+=self_bleu['precisions'][0]
            _in_self_bleu_bi+=self_bleu['precisions'][1]
            _in_self_bleu_tri+=self_bleu['precisions'][2]
            _in_self_bleu_four+=self_bleu['precisions'][3]
            _in_self_bleu_fif+=self_bleu['precisions'][4]
            """
    
        in_self_bleu_one+=_in_self_bleu_one/len(one_label)
        in_self_bleu_bi+=_in_self_bleu_bi/len(one_label)
        in_self_bleu_tri+=_in_self_bleu_tri/len(one_label)
        in_self_bleu_four+=_in_self_bleu_four/len(one_label)
        in_self_bleu_fif+=_in_self_bleu_fif/len(one_label)
        
        _in_self_bleu_one=0
        _in_self_bleu_bi=0
        _in_self_bleu_tri=0
        _in_self_bleu_four=0
        _in_self_bleu_fif=0


        for j in range(len(one_fake)):
            except_one_fake=one_fake[0:j]+one_fake[j+1:]
            refs=[]
            for predicts in except_one_fake:
                refs.append(tweet_tokenizer.tokenize(predicts))
            #print(refs)
            hyp=tweet_tokenizer.tokenize(one_fake[j])
            #print(hyp)
            self_bleu=bleu.sentence_bleu(refs,hyp,weights=[(1./2.,1./2.),(1./3.,1./3.,1./3.),(1./4.,1./4.,1./4.,1./4.),(1./5.,1./5.,1./5.,1./5.,1./5.)])
            #print(self_bleu)
            _in_self_bleu_bi+=self_bleu[0]
            _in_self_bleu_tri+=self_bleu[1]
            _in_self_bleu_four+=self_bleu[2]
            _in_self_bleu_fif+=self_bleu[3]
            """
            self_bleu=_bleu.compute(predictions=[_one_label[j]],references=[except_one_label],max_order=5)
            _in_self_bleu_one+=self_bleu['precisions'][0]
            _in_self_bleu_bi+=self_bleu['precisions'][1]
            _in_self_bleu_tri+=self_bleu['precisions'][2]
            _in_self_bleu_four+=self_bleu['precisions'][3]
            _in_self_bleu_fif+=self_bleu['precisions'][4]
            """

        fake_in_self_bleu_one+=_in_self_bleu_one/len(one_label)
        fake_in_self_bleu_bi+=_in_self_bleu_bi/len(one_label)
        fake_in_self_bleu_tri+=_in_self_bleu_tri/len(one_label)
        fake_in_self_bleu_four+=_in_self_bleu_four/len(one_label)
        fake_in_self_bleu_fif+=_in_self_bleu_fif/len(one_label)
        
        whole_num+=1
        one_label=[]
        one_fake=[]

print("whole texts")
print(whole_num)
print("whole paragraphs")
print(len(whole_predictions))
print(len(whole_labels))
__document_fake=[]
__document_labels=[]
__whole_predictions_len=[]
__whole_labels_len=[]
for j,pred in enumerate(document_fake):
    if len(pred)!=0 and len(document_label[j])!=0:
        __document_fake.append(pred)
        __document_labels.append(whole_labels[j])
        __whole_predictions_len.append(whole_predictions_len[j])
        __whole_labels_len.append(whole_labels_len[j])
print("after remove empty hyp or ref")
print(len(__document_fake))
print(len(__document_labels))
result=rouge.get_scores(__document_fake, __document_labels,avg=True)
# __whole_predictions=[]
# __whole_labels=[]
# __whole_predictions_len=[]
# __whole_labels_len=[]
# for j,pred in enumerate(whole_predictions):
#     if len(pred)!=0 and len(whole_labels[j])!=0:
#         __whole_predictions.append(pred)
#         __whole_labels.append(whole_labels[j])
#         __whole_predictions_len.append(whole_predictions_len[j])
#         __whole_labels_len.append(whole_labels_len[j])
# print("after remove empty hyp or ref")
# print(len(__whole_predictions))
# print(len(__whole_labels))
# result=rouge.get_scores(__whole_predictions, __whole_labels,avg=True)
print(result)



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
fake_in_self_bleu_one=fake_in_self_bleu_one/whole_num
fake_in_self_bleu_bi=fake_in_self_bleu_bi/whole_num
fake_in_self_bleu_tri=fake_in_self_bleu_tri/whole_num
fake_in_self_bleu_four=fake_in_self_bleu_four/whole_num
fake_in_self_bleu_fif=fake_in_self_bleu_fif/whole_num
print("fake_in_self_bleu_one : " + str(fake_in_self_bleu_one))
print("fake_in_self_bleu_bi : " + str(fake_in_self_bleu_bi))
print("fake_in_self_bleu_tri : " + str(fake_in_self_bleu_tri))
print("fake_in_self_bleu_four : " + str(fake_in_self_bleu_four))
print("fake_in_self_bleu_fif : " + str(fake_in_self_bleu_fif))
print(result['rouge-1']['f'])
print(result['rouge-1']['p'])
print(result['rouge-1']['r'])
print(result['rouge-2']['f'])
print(result['rouge-2']['p'])
print(result['rouge-2']['r'])
print(result['rouge-l']['f'])
print(result['rouge-l']['p'])
print(result['rouge-l']['r'])
print("prediction length")
print(np.mean(np.array(__whole_predictions_len)))
print("labels lenght")
print(np.mean(np.array(__whole_labels_len)))

    
# total_results=[in_self_bleu_one,in_self_bleu_bi,in_self_bleu_tri,in_self_bleu_four,in_self_bleu_fif,fake_in_self_bleu_one,fake_in_self_bleu_bi,fake_in_self_bleu_tri,fake_in_self_bleu_four,fake_in_self_bleu_fif,
#                result['rouge-1']['f'],result['rouge-1']['p'],result['rouge-1']['r'],result['rouge-2']['f'],result['rouge-2']['p'],result['rouge-2']['r'],result['rouge-l']['f'],result['rouge-l']['p'],result['rouge-l']['r'],
#                np.mean(np.array(__whole_predictions_len)),np.mean(np.array(__whole_labels_len))]
# total_results_name=["in_self_bleu_one","in_self_bleu_bi","in_self_bleu_tri","in_self_bleu_four","in_self_bleu_fif","fake_in_self_bleu_one","fake_in_self_bleu_bi","fake_in_self_bleu_tri","fake_in_self_bleu_four","fake_in_self_bleu_fif",
#                "result['rouge-1']['f']","result['rouge-1']['p']","result['rouge-1']['r']","result['rouge-2']['f']","result['rouge-2']['p']","result['rouge-2']['r']","result['rouge-l']['f']","result['rouge-l']['p']","result['rouge-l']['r']",
#                "np.mean(np.array(__whole_predictions_len))","np.mean(np.array(__whole_labels_len))"]
total_results=[in_self_bleu_one,in_self_bleu_bi,in_self_bleu_tri,in_self_bleu_four,in_self_bleu_fif,fake_in_self_bleu_one,fake_in_self_bleu_bi,fake_in_self_bleu_tri,fake_in_self_bleu_four,fake_in_self_bleu_fif,
               result['rouge-1']['f'],result['rouge-2']['f'],result['rouge-l']['f'],
               np.mean(np.array(__whole_predictions_len)),np.mean(np.array(__whole_labels_len))]
total_results_name=["in_self_bleu_one","in_self_bleu_bi","in_self_bleu_tri","in_self_bleu_four","in_self_bleu_fif","fake_in_self_bleu_one","fake_in_self_bleu_bi","fake_in_self_bleu_tri","fake_in_self_bleu_four","fake_in_self_bleu_fif",
               "result['rouge-1']['f']","result['rouge-2']['f']","result['rouge-l']['f']",
               "np.mean(np.array(__whole_predictions_len))","np.mean(np.array(__whole_labels_len))"]



file_name = testfile_name+'.txt'

with open(file_name, 'w+') as file:
    for i, val in enumerate(total_results):
        file.write(total_results_name[i] + " : " + str(val) + '\n')
