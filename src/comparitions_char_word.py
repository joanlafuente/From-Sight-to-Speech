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

from Models.bl_gru_arch import Bl_gru
# from data_utils import create_dataset # esta al init de data_utils
from data_utils.utils import *
from data_utils.dataset import *

unique_chars = ['<SOS>', '<EOS>', '<PAD>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_WORDS = len(unique_chars)
idx2word = {k: v for k, v in enumerate(unique_chars)}
word2idx = {v: k for k, v in enumerate(unique_chars)}
TOTAL_MAX_WORDS = 201

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

model = Bl_gru(text_max_len=TOTAL_MAX_WORDS, NUM_CHARS=NUM_WORDS, char2idx=word2idx, idx2char = None, teacher_forcing_ratio=0, rnn_layers=1).to(device)


model.load_state_dict(torch.load(f'/fhome/gia07/Image_captioning/src/runs-baseline/ablation_study_setups/baseline_char_final_resnet_notfreeze/weights/99.pth', map_location=device))

model.eval()

transform2use = T.Compose([v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transformToVisualize = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                        std=[1/0.229, 1/0.224, 1/0.255])])

name_img = "skiing"

img_path = f'/fhome/gia07/Image_captioning/Imgs_test/Imgs/{name_img}.jpg'
img2plot = Image.open(img_path).convert('RGB')
img = transform2use(img2plot)
img = img.to(device)
img = img.unsqueeze(0)

outputs = model(img)


_, preds = torch.max(outputs, dim=1)
preds = preds.cpu().numpy()

pred_text = ''.join([idx2word[word] for word in preds[0] if word not in (0, 1, 2)])

print("Caption:")
print(pred_text)

plt.imshow(img2plot)
plt.title(pred_text)
plt.axis("off")
plt.savefig(f'/fhome/gia07/Image_captioning/Imgs_test/Captions/{name_img}_captioned.png')