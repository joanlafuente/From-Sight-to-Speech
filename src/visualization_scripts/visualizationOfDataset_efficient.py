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

from Models.cross_attention_lstm_correct_efficient_b3_arch import cross_attention_lstm_correct_efficient_b3
# from data_utils import create_dataset # esta al init de data_utils
from data_utils.utils import *
from data_utils.dataset import *


base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
cap_path = f'{base_path}captions.txt'
data = pd.read_csv(cap_path)

# Unique words in the dataset
unique_words = set()
captions = data.caption.apply(lambda x: x.lower()).values
for i in range(len(data)):
    caption = captions[i]
    caption = caption.split()
    unique_words.update(caption)

NUM_WORDS = len(unique_words)
unique_words = ['<SOS>', '<EOS>', '<PAD>'] + sorted(list(unique_words))
idx2word = {k: v for k, v in enumerate(unique_words)}
word2idx = {v: k for k, v in enumerate(unique_words)}
TOTAL_MAX_WORDS = 38


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
partitions = np.load(path_partitions, allow_pickle=True).item()

model = cross_attention_lstm_correct_efficient_b3(text_max_len=38, teacher_forcing_ratio=0, rnn_layers=1, return_attn=True).to(device)

metric = metrics_evaluation

model.load_state_dict(torch.load(f'/fhome/gia07/Image_captioning/src/runs-baseline/attention_efficientb3_1_cap/weights/99.pth', map_location=device))

model.eval()

dataset_train = Data_word_visuals(data, partitions['train'], train=False, TOTAL_MAX_WORDS=TOTAL_MAX_WORDS, word2idx=word2idx)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size= 1, shuffle= True)
# Val
dataset_valid = Data_word_visuals(data, partitions['valid'], train=False, TOTAL_MAX_WORDS=TOTAL_MAX_WORDS, word2idx=word2idx)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size= 1, shuffle= True)
# Test
dataset_test = Data_word_visuals(data, partitions['test'], train=False, TOTAL_MAX_WORDS=TOTAL_MAX_WORDS, word2idx=word2idx)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size= 1, shuffle= False)
    
transformToVisualize = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                        std=[1/0.229, 1/0.224, 1/0.255])])

for img, caption in tqdm(dataloader_test):
    img = img.to(device)
    cap_idx = caption
    outputs, att_h, att_c = model(img)
    
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


    cap_idx = '\n'.join([' '.join([word[0] for word in cap if word2idx[word[0]] not in (0, 1, 2)]) for cap in cap_idx])

    print("PREDS")
    print(pred_text)
    print()
    print("GT")
    print(cap_idx)

    words = [idx2word[word] for word in preds[0]]
    list_words_plot = []
    attentions_h = []
    attentions_c = [] 
    for i in range(len(words)-1):
        if words[i+1] == '<PAD>':
            break
        list_words_plot.append(words[i+1])
        attentions_h.append(att_h[:, i, :].squeeze(0).cpu().detach().view(7, 7))
        attentions_c.append(att_c[:, i, :].squeeze(0).cpu().detach().view(7, 7))


    fig, axs = plt.subplots(1, max(1, len(list_words_plot)), figsize=(30, 10))
    for i in range(len(list_words_plot)):
        # Interpolate the 7x7 map to 224x224
        att = torch.tensor(attentions_h[i] + attentions_c[i]).view(-1)
        # Pass through softmax
        att = F.softmax(att, dim=0).view(1, 1, 7, 7)
        
        att = F.interpolate(att, size=(224, 224), mode='bilinear').view(224, 224).numpy()
        
        axs[i].imshow(img2plot)
        axs[i].imshow(att, alpha=0.5, cmap='jet')
        axs[i].set_title(list_words_plot[i])
        axs[i].axis("off")

        
    plt.axis("off")
    plt.savefig('attentionsTest.png')
    plt.close()
    if input() == 'q':
        break
    