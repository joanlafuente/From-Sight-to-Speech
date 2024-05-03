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

from Models.VIT_transformer_updated_pretrained_linear_arch import VIT_transformer_updated_pretrained_linear
# from data_utils import create_dataset # esta al init de data_utils
from data_utils.utils import *
from data_utils.dataset import *



with open('/fhome/gia07/Image_captioning/src/runs-baseline/attention_efficientb3_multiple_layers_1_cap/word2idx.json', 'r') as f:
    word2idx = json.load(f)

idx2word = {v: k for k, v in word2idx.items()}
NUM_WORDS = len(word2idx)

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

model = VIT_transformer_updated_pretrained_linear(text_max_len=40, NUM_WORDS=NUM_WORDS, teacher_forcing_ratio=0, word2idx=word2idx, transformer_heads=12, transformer_layers=5, dropout=0, pretrained_embedding = False).to(device)


model.load_state_dict(torch.load(f'/fhome/gia07/Image_captioning/src/runs-baseline/VIT_pretrained_emb_lowerlr_5cap_evenSimpler_harder_frozen/best_bleu2.pth', map_location=device))

model.eval()

transform2use = T.Compose([v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transformToVisualize = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                        std=[1/0.229, 1/0.224, 1/0.255])])

name_img = "maya"

img_path = f'/fhome/gia07/Image_captioning/Imgs_test/Imgs/{name_img}.jpg'
img2plot = Image.open(img_path).convert('RGB')
img = transform2use(img2plot)
img = img.to(device)
img = img.unsqueeze(0)

# outputs, attn_h = model(img)
outputs = model(img)

_, preds = torch.max(outputs, dim=1)
preds = preds.cpu().numpy()

pred_text = ' '.join([idx2word[word] for word in preds[0] if word not in (0, 1, 2)])

print("Caption:")
print(pred_text)

plt.imshow(img2plot)
plt.title(pred_text)
plt.axis("off")
plt.savefig(f'/fhome/gia07/Image_captioning/Imgs_test/Captions/{name_img}_captioned.png')


# words = [idx2word[word] for word in preds[0]]
# list_words_plot = []
# attentions_h = []
# for i in range(len(words)-1):
#     if not words[i+1] == '<PAD>':
#         list_words_plot.append(words[i+1])
#         attentions_h.append(att_h[i, 0, :].squeeze(0).cpu().detach().view(7, 7))

# fig, axs = plt.subplots(1, len(list_words_plot)+1, figsize=(35, 10))
# img2plot = transformToVisualize(img)
# img2plot = img2plot.squeeze(0).permute(1, 2, 0).cpu().numpy()
# img2plot = np.clip(img2plot, 0, 1)
# axs[0].imshow(img2plot)
# axs[0].set_title("Original image")
# axs[0].axis("off")
# for i in range(len(list_words_plot)):
#     # Interpolate the 7x7 map to 224x224
#     att = F.interpolate(attentions_h[i].unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear').view(224, 224).numpy()
    
#     axs[i+1].imshow(img2plot)
#     axs[i+1].imshow(att, alpha=0.35, cmap='jet')
#     axs[i+1].set_title(list_words_plot[i])
#     axs[i+1].axis("off")
    
# plt.axis("off")
# plt.show()
# plt.savefig(f'/fhome/gia07/Image_captioning/Imgs_test/Attentions/{name_img}_attentions.png', dpi=300)
# plt.close()
