import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
current_device = torch.cuda.current_device()
print(f"Using GPU: {torch.cuda.get_device_name(current_device)}")
import eyeclip
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from tqdm import tqdm
import logging
import time
from datetime import datetime
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import open_clip


model, preprocess = eyeclip.load("ViT-B/32",device=device,jit=False)
#checkpoint = torch.load("./ft_checkpoints/CLIP_ft_12-06-0119/epoch24.pt")

#model.load_state_dict(checkpoint['model_state_dict'], strict=False)

#imgbank
import pandas as pd
csvp = './multilabel/imagebank2024/dis4ywf_100.csv'
col='CWF'
df=pd.read_csv(csvp)#.sample(500)
p='/home/danli/data/public/imagebank2024/crop/'
df['impath']=p+df['imid']
df.head()

images = torch.cat([preprocess(Image.open(i)).unsqueeze(0) for i in df['impath'].to_list()]).to(device)

chunk_size = 50
top1_recall = 0.0
top5_recall = 0.0
top10_recall = 0.0
total_queries = 0

t_list = df[col].to_list()
for t in t_list:
    embeded_t = eyeclip.tokenize(t, truncate=True).to(device)
    #embeded_t = tokenizer(t, context_length=256).to(device)
    #print(embeded_t.shape)
    text_feature = model.encode_text(embeded_t).to(device)
    text_feature  /= text_feature.norm(dim=-1, keepdim=True)
    all_values = []
    all_indices = []
    #print(text_feature.shape)
    for i in range(0, len(images), chunk_size):
        image_chunk = images[i:i+chunk_size]
        with torch.no_grad():
            image_feature = model.encode_image(image_chunk)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)

        similarity = (100.0 * text_feature @ image_feature.T).softmax(dim=-1)
        if len(similarity)<10:
           values, indices = similarity[0].topk(len(similarity)) 
        else:
            values, indices = similarity[0].topk(10)
        all_values.append(values)
        all_indices.append(indices + i)  
        del image_feature
        del similarity
        torch.cuda.empty_cache()

    all_values = torch.cat(all_values, dim=-1)
    all_indices = torch.cat(all_indices, dim=-1)

    final_values, final_indices = all_values.topk(10)
    final_indices = all_indices[final_indices]
    top1_found = any(t_list[index.item()]==t for index in final_indices[:1])
    top5_found = any(t_list[index.item()]==t for index in final_indices[:5])
    top10_found = any(t_list[index.item()]==t for index in final_indices[:10])

    top1_recall += 1 if top1_found else 0
    top5_recall += 1 if top5_found else 0
    top10_recall += 1 if top10_found else 0

    total_queries += 1
    if total_queries %10 ==0:
        print(total_queries)

    del text_feature
    del all_values
    del all_indices
    torch.cuda.empty_cache()
    #break

top1_recall /= total_queries
top5_recall /= total_queries
top10_recall /= total_queries

print(f"Top1 Recall: {top1_recall}")
print(f"Top5 Recall: {top5_recall}")
print(f"Top10 Recall: {top10_recall}")
