import csv
import ctypes as ct
import pickle
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
from tqdm import tqdm, trange
from rake_nltk import Rake


t_v_t='valid'
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


with open(t_v_t+"_reedsy_prompts_whole.pickle","rb") as fi:
    reedsy = pickle.load(fi)


file=t_v_t+"_reedsy_rake"
f = open(file,'w', newline='')
wr = csv.writer(f)
#fi = open(t_v_t+"_reedsy_prompts.source",'w', newline='\n')


r = Rake()
topK=30

for line in tqdm(reedsy):
    story=line['story'].strip()
    prompt=line['prompt'].strip()
    r.extract_keywords_from_text(story)
    top_features = r.get_ranked_phrases()
    top_features = clean_top_features(top_features, topK)
    keywordsSTR = convert_keys_to_str(top_features)
    
    if len(top_features)==0:
        print("error")
        print(story)
        print(prompt)
    wr.writerow([story,keywordsSTR,prompt])
    
    #print(keywordsSTR)
    #input()
    #f.write(story+'\n')

    #fi.write(prompt+'\n')
    
