from tqdm import trange, tqdm
import pandas as pd
import pickle
import operator

with open("reedsy_prompts.pickle","rb") as fi:
        reedsy = pickle.load(fi)

sorted_reedsy = sorted(reedsy, key=(lambda x: x['name']))

print("before. "+ str(len(sorted_reedsy)))

for i,line in (enumerate(tqdm(sorted_reedsy))):
        while True:
            if (i+1)<len(sorted_reedsy) and line['prompt'] == sorted_reedsy[i+1]['prompt'] and line['name'] == sorted_reedsy[i+1]['name']:
                print("deletion 수행.")
                sorted_reedsy.pop(i+1)
            else:
                break

print("after deletion . " + str(len(sorted_reedsy)))