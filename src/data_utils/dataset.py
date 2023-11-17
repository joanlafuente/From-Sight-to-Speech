import pandas as pd
import numpy as np
import random
from transformers import ResNetModel
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torch

base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
img_path = f'{base_path}Images/'
cap_path = f'{base_path}captions.txt'
path_partitions = f'{base_path}flickr8k_partitions.npy'
data = pd.read_csv(cap_path)

partitions = np.load(path_partitions, allow_pickle=True).item()
chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}
TEXT_MAX_LEN = 201
DEVICE = 'cuda'

class Data_char(Dataset):
    def __init__(self, data, partition):
        self.data = data
        self.partition = partition
        self.num_captions = 5
        self.max_len = TEXT_MAX_LEN
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)

    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        real_idx = self.num_captions*self.partition[idx]
        item = self.data.iloc[real_idx: real_idx+self.num_captions]
        ## image processing
        img_name = item.image.reset_index(drop=True)[0]
        img = Image.open(f'{img_path}{img_name}').convert('RGB')
        img = self.img_proc(img)
    
        ## caption processing
        caption = item.caption.reset_index(drop=True)[random.choice(list(range(self.num_captions)))]
        cap_list = list(caption)
        final_list = [chars[0]]
        final_list.extend(cap_list)
        final_list.extend([chars[1]])
        gap = self.max_len - len(final_list)
        final_list.extend([chars[2]]*gap)
        cap_idx = [char2idx[i] for i in final_list]
        return img, cap_idx

class Data_stats(Dataset):
    def __init__(self, data, partition):
        self.data = data
        self.partition = partition
        self.num_captions = 5
        self.max_len = TEXT_MAX_LEN
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            #v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)

    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        real_idx = self.num_captions*self.partition[idx]
        item = self.data.iloc[real_idx: real_idx+self.num_captions]
        ## image processing
        img_name = item.image.reset_index(drop=True)[0]
        img = Image.open(f'{img_path}{img_name}').convert('RGB')
        img = self.img_proc(img)
    
        ## caption processing
        caption = item.caption.reset_index(drop=True)[random.choice(list(range(self.num_captions)))]
        cap_list = list(caption)
        final_list = cap_list
        # final_list.extend(cap_list)
        #final_list.extend([chars[1]])
        # gap = self.max_len - len(final_list)
        # final_list.extend([chars[2]]*gap)
        cap_idx = [char2idx[i] for i in final_list]

        # Tokenize captions per word
        list_of_words = []
        for word in caption.split():
            list_of_words.append(word)
        return img, cap_idx, list_of_words
    
def get_loader_stats(partition, config):
    dataset = Data_stats(data, partitions[partition])
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size = config[partition]['batch_size'], 
                                         shuffle    = config[partition]['shuffle'])
    return loader