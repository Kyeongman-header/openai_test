from rake_nltk import Rake
import pandas as pd
from pandas import DataFrame
import nltk
nltk.download('stopwords')
r = Rake()
T="train"

total_target=[]

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
for i in range(len(whole_data)):
    
    r.extract_keywords_from_text(whole_data[i])
    words=''
    for word in r.get_ranked_phrases():
        words+=word + ' '
    dict['target'].append(whole_data[i])
    dict['prompt'].append(total_source[0][i])
    dict['keywords'].append(words)

df=pd.DataFrame.from_dict(dict)
df.to_csv("train_wp_rake_results.csv", index=False)