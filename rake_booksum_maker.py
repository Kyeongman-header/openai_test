from rake_nltk import Rake
import pandas as pd
from pandas import DataFrame
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('stopwords')
r = Rake()

T="train"

def sorting(lst):
    # lst2=sorted(lst, key=len)
    lst2 = sorted(lst, key=len)
    return lst2
def clean_top_features(keywords, top=10):
    keywords = sorting(keywords)
    newkeys = []
    newkeys.append(keywords[len(keywords)-1])
    for i in range(len(keywords)-2,-1,-1):
        if newkeys[len(newkeys)-1].startswith(keywords[i]):
            continue
        newkeys.append(keywords[i])

    if len(newkeys) > top:
        return newkeys[:top]
    return newkeys

def convert_keys_to_str(key_list):
    newstr = key_list[0]
    for k in range(1, len(key_list)):
        if len(key_list[k].split(' ')) > 2 :
            newstr += '[SEP]' + key_list[k]
    return newstr.replace("(M)", "").strip()

import csv
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

f = pd.read_csv("booksum/booksum_"+T+'.csv',chunksize=1000)
f= pd.concat(f)

file=T+"_wp_rake_results.csv"
wr = open(file,'w', newline='')
wr=csv.writer(wr)


first=True

r = Rake()
topK=30

for index, line in f.iterrows()::
    story=line[9].strip()
    prompt=sent_tokenizer(line[9])[0].strip()

    r.extract_keywords_from_text(story)
    top_features = r.get_ranked_phrases()
    top_features = clean_top_features(top_features, topK)
    keywordsSTR = convert_keys_to_str(top_features)
    
    if len(top_features)==0:
        print("error")
        print(story)
        print(prompt)
    wr.writerow([story,keywordsSTR,prompt])



