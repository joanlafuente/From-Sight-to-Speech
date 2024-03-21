from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# from data_utils.dataset import get_loader_visuals

import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from torchvision.transforms import v2
import torch.nn.functional as F
import random
import json

from Models.lstm_convnext_tiny_bahdanau_input_multiple_layers_arch import lstm_convnext_tiny_bahdanau_input_multiple_layers
# from data_utils import create_dataset # esta al init de data_utils
from data_utils.utils import *
from data_utils.dataset import *


base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
cap_path = f'{base_path}captions.txt'
data = pd.read_csv(cap_path)

with open('/fhome/gia07/Image_captioning/src/runs-baseline/attention_efficientb3_multiple_layers_1_cap/word2idx.json', 'r') as f:
    word2idx = json.load(f)

idx2word = {v: k for k, v in word2idx.items()}
NUM_WORDS = len(word2idx)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

partitions = np.load(path_partitions, allow_pickle=True).item()

model = lstm_convnext_tiny_bahdanau_input_multiple_layers(text_max_len=38, NUM_WORDS=NUM_WORDS, word2idx=word2idx, teacher_forcing_ratio=0, rnn_layers=4, return_attn=True).to(device)

metric = metrics_evaluation

model.load_state_dict(torch.load(f'/fhome/gia07/Image_captioning/src/runs-baseline/attention_convnext_tiny_bahdanau_input_multiple_layers_1_cap/weights/72.pth', map_location=device))

model.eval()

dataset_train = Data_word_visuals(data, partitions['train'], TOTAL_MAX_WORDS=38, word2idx=word2idx, train=False)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size= 1, shuffle= True)
# Val
dataset_valid = Data_word_visuals(data, partitions['valid'], TOTAL_MAX_WORDS=38, word2idx=word2idx, train=False)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size= 1, shuffle= True)
# Test
dataset_test = Data_word_visuals(data, partitions['test'], TOTAL_MAX_WORDS=38, word2idx=word2idx, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size= 1, shuffle= False)
    
transformToVisualize = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                        std=[1/0.229, 1/0.224, 1/0.255])])
specific_img_id = 305

for e, (img, caption) in enumerate(tqdm(dataloader_test)):
    if specific_img_id is not None:
        if e != (specific_img_id):
            continue
    img = img.to(device)
    cap_idx = caption
    outputs, att_h = model(img)


    img = transformToVisualize(img)
    img2plot = np.clip(img.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
    plt.imshow(img2plot)
    plt.axis("off")
    plt.savefig('visualsTest.png')

    # val_loss = crit(outputs, cap_idx[:, random.choice(list(range(cap_idx.size(1)))), :].squeeze(1))

    _, preds = torch.max(outputs, dim=1)
    preds = preds.cpu().numpy()
    # print(preds)


    # preds = [' '.join([idx2word[word] for word in pred if word not in (0, 1, 2)])
    #             for pred in preds]
    
    pred_text = ' '.join([idx2word[word] for word in preds[0] if word not in (0, 1, 2)])

    # If word not in vocab, replace with <UNK>
    cap_idx = [[word[0] if word[0] in word2idx.keys() else '<UNK>' for word in cap] for cap in cap_idx]
    cap_idx = '\n'.join([' '.join([word for word in cap if word2idx[word] not in (0, 1, 2)]) for cap in cap_idx])

    print("PREDS")
    print(pred_text)
    print()
    print("GT")
    print(cap_idx)

    words = [idx2word[word] for word in preds[0]]
    list_words_plot = []
    attentions_h = []
    for i in range(len(words)-1):
        if not words[i+1] == '<PAD>':
            list_words_plot.append(words[i+1])
            attentions_h.append(att_h[i, 0, :].squeeze(0).cpu().detach().view(7, 7))

    fig, axs = plt.subplots(1, len(list_words_plot)+1, figsize=(35, 10))
    axs[0].imshow(img2plot)
    axs[0].set_title("Original image")
    axs[0].axis("off")
    for i in range(len(list_words_plot)):
        # Interpolate the 7x7 map to 224x224
        att = F.interpolate(attentions_h[i].unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear').view(224, 224).numpy()
        
        axs[i+1].imshow(img2plot)
        axs[i+1].imshow(att, alpha=0.35, cmap='jet')
        axs[i+1].set_title(list_words_plot[i])
        axs[i+1].axis("off")
        
    plt.axis("off")
    plt.show()
    plt.savefig(f'attentions_imgid{e}.png', dpi=300)
    plt.close()
    if input() == 'q':
        break
    