from rake_nltk import Rake
import pandas as pd
from pandas import DataFrame
import nltk
nltk.download('stopwords')
r = Rake()
T="train"

total_target=[]
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

with open("/home/ubuntu/research/writingPrompts/"+ T +".wp_target", encoding='UTF8') as f:
    stories = f.readlines()
    stories = [" ".join(i.split()[0:1000]) for i in stories]
    temp_stories=[]
    for story in stories:
        temp_stories.append(story.replace("<newline>",""))
    total_target.append(temp_stories)
total_source=[]

with open("/home/ubuntu/research/writingPrompts/"+ T +".wp_source", encoding='UTF8') as f:
    stories = f.readlines()
    stories = [" ".join(i.split()[0:1000]) for i in stories]
    temp_stories=[]
    for story in stories:
        temp_stories.append(story.replace("<newline>",""))
    total_source.append(temp_stories)

RANGE=0
START=0 # 마지막으로 끝난 라인부터 다시 만든다.

if RANGE !=0:
    whole_data=total_target[0][START:START+RANGE]
else:
    whole_data=total_target[0][START:]

dict={'target' : [], 'prompt' : [], 'keyword' : []}
topK=30
import csv
file=T+"_wp_rake_results.csv"
f = open(file,'w', newline='')
wr = csv.writer(f)

for i in range(len(whole_data)):

    story=whole_data[i].strip()
    r.extract_keywords_from_text(story)
    top_features = r.get_ranked_phrases()
    top_features = clean_top_features(top_features, topK)
    keywordsSTR = convert_keys_to_str(top_features)
    if len(top_features)==0:
        print("error")
        print(story)
    
    print(whole_data[i])
    print(keywordsSTR)
    print()
    input()
    wr.writerow([story,keywordsSTR,total_source[0][i]])