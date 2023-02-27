
import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import AutoConfig
from dataset_consts import *

#config = AutoConfig.from_pretrained('./checkpoint-',gradient_checkpointing=True,)
#model =  AutoModelForSeq2SeqLM.from_config(config) # pretrained from myself.
model=AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large',)

from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer


TRAIN_RANGE=25000

# train_total_target=total_target[:TRAIN_RANGE]
# train_total_source=total_source[:TRAIN_RANGE]
# val_total_target=total_target[TRAIN_RANGE:]
# val_total_source=total_source[TRAIN_RANGE:]

# train_dataset=return_dataset(train_total_target,train_total_source)
# valid_dataset=return_dataset(val_total_target,val_total_source)

with open("pickle_data/"+"ROCStories_train"+"/level_1.pickle","rb") as fi:
        train_dataset = pickle.load(fi)
with open("pickle_data/"+"ROCStories_valid"+"/level_1.pickle","rb") as fi:
        valid_dataset = pickle.load(fi)

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

createFolder("logs")

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #fp16=True,
    #fp16_backend="apex",
    output_dir="./",
    logging_steps=100,
    logging_strategy="steps",
    logging_dir="./logs/"+current_time,
    eval_delay=5,
    warmup_steps=1500,
    save_total_limit=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
)



model.config.num_beams = 4
model.config.max_length = 512
model.config.min_length = 100
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()

# model.save_pretrained('./first_level_best.pt')
