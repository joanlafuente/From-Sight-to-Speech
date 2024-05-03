import torch
import json
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
import random
import matplotlib.pyplot as plt

from Models import define_network # esta al init de models
from data_utils import create_dataset # esta al init de data_utils
from data_utils.utils import *
from data_utils.dataset import *

CUDA_LAUNCH_BLOCKING = 1
# TORCH_USE_CUDA_DSA = 1


def generate_captions(model, metric, dataloader, weights=None):
    # model.eval()
    loss = 0
    list_pred = []
    list_ref = []
    id_img = 0
    transform = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                        std=[1/0.229, 1/0.224, 1/0.255])])
    if not os.path.exists(f"{config['root_dir']}{weights}/bleu2"):
        os.makedirs(f"{config['root_dir']}{weights}/bleu2")
    if not os.path.exists(f"{config['root_dir']}{weights}/meteor"):
        os.makedirs(f"{config['root_dir']}{weights}/meteor")
        
    with torch.no_grad():
        for img, cap_idx, regions in tqdm(dataloader):
            batch_size = img.shape[0]
            img = img.to(device)
            cap_idx = cap_idx.to(device)
            outputs = model(img, regions)

            _, preds = torch.max(outputs, dim=1)
            # preds = preds.permute(1, 0)
            preds = preds.cpu().numpy()
            cap_idx = cap_idx.cpu().numpy()
        
            preds = [' '.join([idx2word[word] for word in pred if word not in (0, 1, 2)])
                     for pred in preds]

            cap_idx = [[' '.join([idx2word[word] for word in pred if word not in (0, 1, 2)])
                     for pred in cap] for cap in cap_idx]
            
            list_pred.extend(preds)
            list_ref.extend(cap_idx)
            
            for i in range(batch_size):
                metrics_item = metric([preds[i]], [cap_idx[i]])
                meteor = metrics_item["meteor"]
                meteor = round(meteor, 4)
                bleu2 = metrics_item["bleu2"]
                bleu2 = round(bleu2, 4)
                image = transform(img[i])
                plt.imshow(image.cpu().permute(1, 2, 0))
                plt.title(preds[i])
                plt.axis("off")
                plt.savefig(f"{config['root_dir']}{weights}/bleu2/{id_img}_{bleu2}.png")
                plt.savefig(f"{config['root_dir']}{weights}/meteor/{id_img}_{meteor}.png")
                plt.close()
                id_img += 1
                

    res = metric(list_pred, list_ref)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='Regions_pretrained_emb_lowerlr_5cap_evenSimpler_harder')
    args = parser.parse_args()
    config = LoadConfig_baseline(args.test_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_path = config["datasets"]["base_path"]
    img_path = f'{base_path}Images/'
    cap_path = f'{base_path}captions.txt'
    path_partitions = f'{base_path}flickr8k_partitions.npy'
    data = pd.read_csv(cap_path)

    # Open partitions file
    NUM_CAPTIONS = 5
    partitions = np.load(path_partitions, allow_pickle=True).item()


    with open(f"{config['root_dir']}word2idx.json", "r") as f:
       word2idx = json.load(f)
        
    idx2word = {v: k for k, v in word2idx.items()}
    NUM_WORDS = len(word2idx.keys())


    weights = None

    partitions = np.load(path_partitions, allow_pickle=True).item()
    TOTAL_MAX_WORDS = config['network']['params']['text_max_len']
    
    size = config["datasets"].get('size', 224)
    # Train
    dataset_train = create_dataset(data, partitions['train'], TOTAL_MAX_WORDS, word2idx = word2idx, train=True, augment=config['datasets']['augment_imgs'], dataset_name="TrainSet", dataset_type=config["datasets"]["type"], size=size)# Data_word_aug(data, partitions['train'], train=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, **config["datasets"]["train"])
    # Val
    dataset_valid = create_dataset(data, partitions['valid'], TOTAL_MAX_WORDS, word2idx = word2idx, train=False, augment=False, dataset_name="ValidSet", dataset_type=config["datasets"]["type"], size=size)# Data_word_aug(data, partitions['val'], train=False)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, **config["datasets"]["valid"])
    # Test
    dataset_test = create_dataset(data, partitions['test'], TOTAL_MAX_WORDS, word2idx = word2idx, train=False, augment=False, dataset_name="TestSet", dataset_type=config["datasets"]["type"], size=size)# Data_word_aug(data, partitions['test'], train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **config["datasets"]["test"])
    
    config["network"]["params"]["NUM_WORDS"] = NUM_WORDS
    config["network"]["params"]["word2idx"]  = word2idx
    config["network"]["params"]["idx2word"]  = idx2word
    
    # print("NUM_WORDS", NUM_WORDS)

    model = define_network(config["network"]["params"]) # LSTM_attention(TOTAL_MAX_WORDS, **config["network"]["params"], device = device).to(device)
    
    if config["network"]["checkpoint"] is not None:
        model.load_state_dict(torch.load(model.config["network"]["checkpoint"]))
        print("Loading checkpoint from", model.config["network"]["checkpoint"])
    
    if config["network"].get("save_attention", False) is not False:
        save_output = SaveOutput()
        patch_attention(model.transformer_decoder.layers[-1].self_attn)
        hook_handle = model.transformer_decoder.layers[-1].self_attn.register_forward_hook(save_output)


    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    print(torch.cuda.device_count())
    
    model = model.to(device)
    model.eval()
    
    metric = metrics_evaluation
    metric_earlystop = config["early_stopping"]["metric"]
    start_epoch = 0
    counter = 0
    best_val_metric = 0


    # last_epoch = os.listdir(config["weights_dir"])
    # last_epoch = max([int(epoch.split(".")[0]) for epoch in last_epoch])
    
    # model.load_state_dict(torch.load(f'{config["weights_dir"]}{last_epoch}.pth', map_location=device))
    # model.eval()
    
    # res_t = generate_captions(model, metric, dataloader_test, weights="last_epoch")


    model.load_state_dict(torch.load(f'{config["root_dir"]}best_{metric_earlystop}.pth', map_location=device))
    # model.eval()

    res_t = generate_captions(model, metric, dataloader_test, weights=f"best_{metric_earlystop}_epoch")
