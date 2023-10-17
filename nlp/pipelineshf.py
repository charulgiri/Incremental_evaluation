from datasets import load_dataset
from transformers import AutoTokenizer, BertModel, pipeline
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, OpenAIGPTModel
from transformers import OpenAIGPTForSequenceClassification, BertForSequenceClassification
import torch
import numpy as np
import logging
import time
from tmu.tools import BenchmarkTimer
import pandas as pd

def preprocess_pipeline(batch):
    temp = []
    for example in batch:
        examp = example["text"].split(" ")
        # print(examp)
        if len(examp)> 250:
            examp = " ".join(examp[:250])
        else:
            examp = example["text"]
        temp.append(examp)
    return temp



tokenizer_bert = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ************** load data **************
imdb = load_dataset("imdb")
imdb["test"]=imdb["test"].train_test_split(test_size=0.1)["test"]
# imdb_bert = imdb["test"]
# imdb_gpt = imdb["test"]

imdb_pipeline = preprocess_pipeline(imdb["test"])

# # ************** test bert pipeline **************
pipe_bert = pipeline("text-classification", model="roberta-base")
start_time = time.time()
text = pipe_bert(imdb_pipeline)
end_time = time.time()
print(f"Bert Pipeline\tInference time taken: {(end_time-start_time):.2f}s")