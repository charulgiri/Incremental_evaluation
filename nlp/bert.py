from datasets import load_dataset
from transformers import AutoTokenizer, BertModel, pipeline
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, OpenAIGPTModel
from transformers import OpenAIGPTForSequenceClassification, BertForSequenceClassification
import torch
import numpy as np
# import evaluate
import logging
import time
from tmu.tools import BenchmarkTimer
import pandas as pd

_LOGGER = logging.getLogger(__name__)

tokenizer_bert = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer_gpt  = AutoTokenizer.from_pretrained("openai-gpt")
# if tokenizer_gpt.pad_token is None:
#     tokenizer_gpt.add_special_tokens({'pad_token': '[PAD]'})

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

def preprocess_bert(batch):
    batch["input_ids"] = tokenizer_bert(batch["text"], padding="max_length", return_tensors="pt", truncation=True).input_ids
    return batch

def preprocess_data(data):
    data = data.map(preprocess_bert)["input_ids"]
    data = torch.tensor(data)
    data = torch.squeeze(data, 1)
    return data

def pipelines(model_name, data, dataset_name):
    pipe = pipeline("text-classification", model=model_name)
    start_time = time.time()
    text = pipe(data)
    end_time = start_time - time.time()
    print(f"Pipeline:{model_name}\tDataset:{dataset_name}\tInference time:{end_time:.2f}")

def models(model_name, data, dataset_name, task_type="single_label_classification"):
    print(f"Processing Data....")
    inputs = preprocess_data(data)
    # inputs = inputs.to(device)
    print(f"inputs.is_cuda:{inputs.is_cuda}")
    print(f"Processing Data.... Done!")
    if model_name=="gpt":
        if task_type=="single_label_classification":
            model = OpenAIGPTModel.from_pretrained("openai-gpt")
        else:
            model = OpenAIGPTForSequenceClassification.from_pretrained("openai-gpt", problem_type=task_type)
    elif model_name=="bert":
        if task_type=="single_label_classification":
            id2label = {0: "NEGATIVE", 1: "POSITIVE"}
            label2id = {"NEGATIVE": 0, "POSITIVE": 1}
            model = AutoModelForSequenceClassification.from_pretrained( "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)
        else:
            model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity",  problem_type=task_type)
    else:
        print("Model not implemented")
    # model  = model.to(device)
    start_time = time.time()
    print("Running Inference... ")
    # print(f"model.is_cuda:{model.is_cuda}")
    with torch.no_grad():
        logits = model(inputs)
    end_time =  time.time() - start_time
    print(f"Model:{model_name}\tDataset:{dataset_name}\tInference time:{end_time:.2f}")


# ************** load data **************
imdb = load_dataset("imdb")
imdb_t = imdb["test"].train_test_split(test_size=0.1)["test"]
# tweet_eval = load_dataset("tweet_eval", "emoji", split="test")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
models("gpt", imdb_t, "imdb")


