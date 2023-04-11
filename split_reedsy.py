import pickle
import random

def split(TRAIN_NUM,VALID_NUM,TEST_NUM):
    whole_data=[]
    with open("reedsy_prompts_whole.pickle","rb") as fi:
        reedsy = pickle.load(fi)
    

    START=0
    RANGE=0

    if RANGE !=0:
        whole_data=reedsy[START:START+RANGE]
    else:
        whole_data=reedsy[START:]

    random.shuffle(whole_data)
    train_data=whole_data[:TRAIN_NUM]
    valid_data=whole_data[TRAIN_NUM:TRAIN_NUM+VALID_NUM]
    test_data=whole_data[VALID_NUM:TRAIN_NUM+VALID_NUM+TEST_NUM]
    return train_data,valid_data,test_data

train,valid,test=split(1000,100,100)

with open("train_reedsy_prompts_whole"+".pickle","wb") as f:
        pickle.dump(train,f)
with open("valid_reedsy_prompts_whole"+".pickle","wb") as f:
        pickle.dump(valid,f)
with open("test_reedsy_prompts_whole"+".pickle","wb") as f:
        pickle.dump(test,f)