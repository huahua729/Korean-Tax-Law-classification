from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np
import sys
import re


model_name_or_path = 'KTL-BERT/epoch-1/'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
device = torch.device("cuda:1")
model.to(device)

def first_page(request):
    return render(request, "index.html")


def class_func(request):
    print("----------------------")
    if "POST" == request.method:
        question = request.POST.getlist('data')[0]
        print("question",question)
        if isHangul(question) == True:
            answer = model_func(question)
            if answer == 0:
                class_data = "법인세"
            elif answer == 1:
                class_data = "부가가치세"
            elif answer == 2:
                class_data = "양도소득세"
            elif answer == 3:
                class_data = "원천징수(연말정산)"
            elif answer == 4:
                class_data = "종합소득세"
            return JsonResponse({'class': class_data})
        if isHangul(question) == False:
            return JsonResponse({'class': 0})

def isHangul(text):
    #Check the Python Version
    pyVer3 =  sys.version_info >= (3, 0)

    if pyVer3 : # for Ver 3 or later
        encText = text
    else: # for Ver 2.x
        if type(text) is not unicode:
            encText = text.decode('utf-8')
        else:
            encText = text

    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', encText))
    return hanCount > 0


def model_func(question):
    texts = [question]
    labels = [0]
    batch_encoding = tokenizer.batch_encode_plus(
        [text for text in texts],
        max_length=100,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )

    features = []
    for i in range(len(texts)):
        input = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in input:
            input["token_type_ids"] = [0] * len(input["input_ids"])
        features.append(input)

    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    all_labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=16)

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3]
        }
        inputs["token_type_ids"] = batch[2]
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]

        preds = logits.detach().cpu().numpy()
        preds_new = np.argmax(preds, axis=1)

    return preds_new[0]