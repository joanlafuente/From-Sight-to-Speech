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
import random

from Models import define_network # esta al init de models
from data_utils import create_dataset # esta al init de data_utils
from data_utils.utils import *
from data_utils.dataset import *

CUDA_LAUNCH_BLOCKING = 1
# TORCH_USE_CUDA_DSA = 1

def train_one_epoch(model, optimizer, crit, metric, dataloader):
    loss = 0
    model.train()

    list_pred = []
    list_ref = []
    for i, (img, cap_idx, regions) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        # load it to the active device
        img = img.to(device)
        cap_idx = cap_idx.to(device) # batch, seq
        # Passing the ground truth to the model for teacher forcing
        outputs = model(img, regions = regions) # batch, 80, seq
        print("batch", i, outputs.shape, cap_idx.shape)
        train_loss = crit(outputs, cap_idx)

        _, preds = torch.max(outputs, dim=1)

        # print(preds.shape, cap_idx.shape)
        preds = preds.cpu().numpy()
        cap_idx = cap_idx.cpu().numpy()

        preds = [' '.join([idx2word[word] for word in pred if word not in (0, 1, 2)])
                 for pred in preds]
        
        cap_idx = [' '.join([idx2word[word] for word in pred if word not in (0, 1, 2)])
                 for pred in cap_idx]
        
        list_pred.extend(preds)
        list_ref.extend(cap_idx)
        

        if config["loss"].get("type", None) == "SentEmb_loss":
            # print(list_pred, list_ref)
            train_loss = crit(list_pred, list_ref)
            
            
        train_loss.backward()
        optimizer.step()
        
            # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

        wandb.log({"Train Loss": train_loss.item()}, step = len(dataloader)*len(img)*(epoch)+i*len(img))

    res = metric(list_pred, list_ref)
    os.makedirs(f"{config['train_output_dir']}Epoch_{epoch}", exist_ok=True)
    # Trransform to do an inverse normalitzation of the image
    transform = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                        std=[1/0.229, 1/0.224, 1/0.255])])
    for i in range(5):
        try:
            image = img[-i].cpu()
            # De normalize the image
            image = transform(image)
            pred = list_pred[-i]
            save_image(image, f'{config["train_output_dir"]}Epoch_{epoch}/Caption_{pred}_{i}.png')
        except:
            print("Error saving image")
    return loss/len(dataloader), res

def eval_epoch(model, crit, metric, dataloader, epoch=0):
    model.eval()
    loss = 0
    list_pred = []
    list_ref = []
    with torch.no_grad():
        for img, cap_idx, regions in tqdm(dataloader):
            img = img.to(device)
            cap_idx = cap_idx.to(device)
            outputs = model(img, regions= regions)

            if config["loss"].get("type", None) != "SentEmb_loss":            
                val_loss = crit(outputs, cap_idx[:, random.choice(list(range(cap_idx.size(1)))), :].squeeze(1))
                loss += val_loss.item()

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
            
            if config["loss"].get("type", None) == "SentEmb_loss":
                val_loss = crit(list_pred, list_ref)
                loss += val_loss.item()

    res = metric(list_pred, list_ref)
    if epoch > config["network"]["epoch2unfreeze"]:
        print("Unfreezing the model")
        model.unfreeze_params(True)

    # Save the last 5 images and the predicted captions
    os.makedirs(f"{config['output_dir']}Epoch_{epoch}", exist_ok=True)
    # Trransform to do an inverse normalitzation of the image
    transform = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                        std=[1/0.229, 1/0.224, 1/0.255])])
    for i in range(5):
        try:
            image = img[-i].cpu()
            # De normalize the image
            image = transform(image)
            pred = list_pred[-i]
            save_image(image, f'{config["output_dir"]}Epoch_{epoch}/Caption_{pred}_{i}.png')
        except:
            print("Error saving image")
    
    return loss/len(dataloader), res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='test_attention')
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

    s = []
    for idx in partitions['train']:
        s.extend([(idx * NUM_CAPTIONS) + i for i in range(5)])
    idx = np.array(s)
    result = data.iloc[idx]


    captions = result.caption.apply(lambda x: x.lower()).values

    unique_words = set()
    list_words = []
    for i in range(len(captions)):
        caption = captions[i]
        caption = caption.split()
        list_words.append(caption)
        list_words.append(['<SOS>', '<EOS>'] + ['<PAD>']*(38-len(caption)))
        unique_words.update(caption)

    # Count the number of times that each word appears in the dataset
    word_count = {}
    from collections import Counter
    word_count = Counter([word for caption in list_words for word in caption])
    word_count['<UNK>'] = 1
    # total/(num_clases * count_word)

    total = sum(word_count.values())
    
    unique_words = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + sorted(list(unique_words))
    
    num_classes = len(unique_words)
    NUM_WORDS = len(unique_words)
    idx2word = {k: v for k, v in enumerate(unique_words)}
    word2idx = {v: k for k, v in enumerate(unique_words)}

    word_weights = {word2idx[word]: total/(num_classes * count_word) for word, count_word in word_count.items()}

    weights = torch.tensor([weight for _, weight in sorted(word_weights.items(), key=lambda x: x[0])])

    # Unique words in the dataset
    #unique_words = set()
    #captions = data.caption.apply(lambda x: x.lower()).values
    #for i in range(len(data)):
    #    caption = captions[i]
    #    caption = caption.split()
    #    unique_words.update(caption)
#
    #unique_words = ['<SOS>', '<EOS>', '<PAD>'] + sorted(list(unique_words))
    #NUM_WORDS = len(unique_words)
    #idx2word = {k: v for k, v in enumerate(unique_words)}
    #word2idx = {v: k for k, v in enumerate(unique_words)}
    #TOTAL_MAX_WORDS = 38

    partitions = np.load(path_partitions, allow_pickle=True).item()
    TOTAL_MAX_WORDS = config['network']['params']['text_max_len']
    # Train
    dataset_train = create_dataset(data, partitions['train'], TOTAL_MAX_WORDS, word2idx = word2idx, train=True, augment=config['datasets']['augment_imgs'], dataset_name="TrainSet", dataset_type=config["datasets"]["type"], device = device)# Data_word_aug(data, partitions['train'], train=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, **config["datasets"]["train"])
    # Val
    dataset_valid = create_dataset(data, partitions['valid'], TOTAL_MAX_WORDS, word2idx = word2idx, train=False, augment=False, dataset_name="ValidSet", dataset_type=config["datasets"]["type"], device = device)# Data_word_aug(data, partitions['val'], train=False)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, **config["datasets"]["valid"])
    # Test
    dataset_test = create_dataset(data, partitions['test'], TOTAL_MAX_WORDS, word2idx = word2idx, train=False, augment=False, dataset_name="TestSet", dataset_type=config["datasets"]["type"])# Data_word_aug(data, partitions['test'], train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **config["datasets"]["test"])
    
    config["network"]["params"]["NUM_WORDS"] = NUM_WORDS
    config["network"]["params"]["word2idx"]  = word2idx
    config["network"]["params"]["idx2word"]  = idx2word
    
    # print("NUM_WORDS", NUM_WORDS)

    model = define_network(config["network"]["params"]) # LSTM_attention(TOTAL_MAX_WORDS, **config["network"]["params"], device = device).to(device)
    
    if config["network"]["freeze_encoder"]:
        print("Freezing the model")
        model.unfreeze_params(False)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    print(torch.cuda.device_count())
    
    model = model.to(device)
    model.train()
    optimizer = get_optimer(config["optimizer"], model) # torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = get_scheduler(config["scheduler"], optimizer)
    
    crit = get_loss(config["loss"], weights=weights, device = device)
    metric = metrics_evaluation
    metric_earlystop = config["early_stopping"]["metric"]
    start_epoch = 0
    counter = 0
    best_val_metric = 0

    wandb.init(project='Image_Captioning2', config=config, name=args.test_name)
    wandb.watch(model)
    for epoch in range(start_epoch, config["epochs"]):
        loss, res = train_one_epoch(model, optimizer, crit, metric, dataloader_train)
        print(f'train loss: {loss:.2f}, epoch: {epoch}')
        loss_v, res_v = eval_epoch(model, crit, metric, dataloader_valid, epoch=epoch)
        print(f'valid loss: {loss_v:.2f}')
        wandb.log({"Epoch Train Loss": loss, "Epoch Validation Loss": loss_v, "epoch":epoch+1}, step=(epoch+1)*len(dataloader_train)*config["datasets"]["train"]['batch_size'])
        wandb.log({"Train": res, "Validation": res_v, "epoch":epoch+1}, step=(epoch+1)*len(dataloader_train)*config["datasets"]["train"]['batch_size'])
        scheduler.step() if scheduler is not None else None
        counter += 1
        if epoch % config["network"].get("save_ckpt_every", 3) == 0:
            torch.save(model.state_dict(), f'{config["weights_dir"]}/{epoch}.pth')    
        
        res_v["loss"] = loss_v # aixo es per poder canviar la metrica de early stopping desde el config

        if res_v[metric_earlystop] > best_val_metric:
            counter = 0
            best_val_metric = res_v[metric_earlystop]
            print(f"New best validation {metric_earlystop}")
            torch.save(model.state_dict(), f'{config["root_dir"]}/best_{metric_earlystop}.pth') 
        
        if counter >= config["early_stopping"]["patience"]:
            break

    model.load_state_dict(torch.load(f'{config["root_dir"]}best_{metric_earlystop}.pth'))
    model.eval()
    loss_t, res_t = eval_epoch(model, crit, metric, dataloader_test, epoch="test")
    wandb.log({"Test Loss": loss_t, "Test": res_t}, step=(epoch+1)*len(dataloader_train)*config["datasets"]["train"]['batch_size'])
    print(f'Test loss: {loss_t:.2f}')
    print(f'Test: {res_t}')

    wandb.finish()