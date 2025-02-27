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


from data_utils.dataset import Data_word, Data_word_aug
from data_utils.utils import LoadConfig, metrics_evaluation, get_scheduler, get_weights
# from Models.freeze_resnet import Teacher_img_to_word_LSTM
from Models.lstm_attention2 import LSTM_attention

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

unique_words = ['<SOS>', '<EOS>', '<PAD>'] + sorted(list(unique_words))
NUM_WORDS = len(unique_words)
idx2word = {k: v for k, v in enumerate(unique_words)}
word2idx = {v: k for k, v in enumerate(unique_words)}
TOTAL_MAX_WORDS = 38

def train_one_epoch(model, optimizer, crit, metric, dataloader):
    loss = 0
    model.train()

    list_pred = []
    list_ref = []
    for i, (img, cap_idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # load it to the active device
        img = img.to(device)
        cap_idx = cap_idx.to(device) # batch, seq
        # Passing the ground truth to the model for teacher forcing
        outputs = model(img, cap_idx) # batch, 80, seq
        # print("batch", i, outputs.shape, cap_idx.shape)
        train_loss = crit(outputs, cap_idx)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

        wandb.log({"Train Loss": train_loss.item()}, step = len(dataloader)*len(img)*(epoch)+i*len(img))
        
        _, preds = torch.max(outputs, dim=1)

        preds = preds.cpu().numpy()
        cap_idx = cap_idx.cpu().numpy()

        preds = [' '.join([idx2word[word] for word in pred if word not in (0, 1, 2)])
                 for pred in preds]
        
        cap_idx = [' '.join([idx2word[word] for word in pred if word not in (0, 1, 2)])
                 for pred in cap_idx]
        
        list_pred.extend(preds)
        list_ref.extend(cap_idx)

    res = metrics_evaluation(list_pred, list_ref)
    os.makedirs(f"{config['train_output_dir']}Epoch_{epoch}", exist_ok=True)
    # Trransform to do an inverse normalitzation of the image
    transform = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                        std=[1/0.229, 1/0.224, 1/0.255])])
    for i in range(5):
        image = img[-i].cpu()
        # De normalize the image
        image = transform(image)
        pred = list_pred[-i]
        try:
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
        for img, cap_idx in tqdm(dataloader):
            img = img.to(device)
            cap_idx = cap_idx.to(device)
            outputs = model(img)

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

    res = metrics_evaluation(list_pred, list_ref)
    if epoch == config.get('epoch_freeze', None):
        print("Unfreeze the model")
        model.unfreeze_params(True)
    # Save the last 5 images and the predicted captions
    os.makedirs(f"{config['output_dir']}Epoch_{epoch}", exist_ok=True)
    # Trransform to do an inverse normalitzation of the image
    transform = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                        std=[1/0.229, 1/0.224, 1/0.255])])
    for i in range(5):
        image = img[-i].cpu()
        # De normalize the image
        image = transform(image)
        pred = list_pred[-i]
        try:
            save_image(image, f'{config["output_dir"]}Epoch_{epoch}/Caption_{pred}_{i}.png')
        except:
            print("Error saving image")
    
    return loss/len(dataloader), res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='test_attention')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
    img_path = f'{base_path}Images/'
    cap_path = f'{base_path}captions.txt'
    path_partitions = f'{base_path}flickr8k_partitions.npy'
    data = pd.read_csv(cap_path)

    best_val_metric = 0

    partitions = np.load(path_partitions, allow_pickle=True).item()

    # partitions['train'] = partitions['train'][:int(len(partitions['train'])*0.005)]
    # partitions['valid'] = partitions['valid'][:int(len(partitions['valid'])*0.005)]
    # partitions['test'] = partitions['test'][:int(len(partitions['test'])*0.005)]
    
    metric = metrics_evaluation

    # Train
    dataset_train = Data_word_aug(data, partitions['train'], train=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, **config["datasets"]["train"])
    # Val
    dataset_valid = Data_word_aug(data, partitions['valid'], train=False)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, **config["datasets"]["valid"])
    # Test
    dataset_test = Data_word_aug(data, partitions['test'], train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **config["datasets"]["test"])
    
    model = LSTM_attention(TOTAL_MAX_WORDS, **config["network"]["params"], device = device).to(device)
    
    
    if config.get("freeze", None):
        print("Freeze the model")
        model.unfreeze_params(False)
    model = nn.DataParallel(model)
    # load checkpoint
    # path = get_weights(config["weights_dir"])
    # if path is not None:
    # weights = os.listdir(config["weights_dir"])
    
    # if len(weights) > 0:
    #     epoch_best = max([int(i.split('.')[0]) for i in weights])
    #     model.load_state_dict(torch.load(f'{config["weights_dir"]}/{epoch_best}.pth'))

    #     # start_epoch = int(path.split('/')[-1].split('.')[0])
    #     start_epoch = epoch_best
    #     if config.get("wandb", False) and config["wandb"].get("resume", False):
    #         wandb.init(project='Image_Captioning', config=config, name=args.test_name, resume=config["wandb"]["resume"], id=config["wandb"]["id"]) 
    #     else:
    #         wandb.init(project='Image_Captioning', config=config, name=args.test_name)
    # else: 
    start_epoch = 0
    wandb.init(project='Image_Captioning', config=config, name=args.test_name)

    wandb.watch(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    crit = nn.CrossEntropyLoss()
    metric = "metrics_evaluation"
    counter = 0
    squeduler = get_scheduler(config, optimizer)
    for epoch in range(start_epoch, config["epochs"]):
        loss, res = train_one_epoch(model, optimizer, crit, metric, dataloader_train)
        print(f'train loss: {loss:.2f}, epoch: {epoch}')
        loss_v, res_v = eval_epoch(model, crit, metric, dataloader_valid, epoch=epoch)
        print(f'valid loss: {loss_v:.2f}')
        wandb.log({"Epoch Train Loss": loss, "Epoch Validation Loss": loss_v, "epoch":epoch+1}, step=(epoch+1)*len(dataloader_train)*config["datasets"]["train"]['batch_size'])
        wandb.log({"Train": res, "Validation": res_v, "epoch":epoch+1}, step=(epoch+1)*len(dataloader_train)*config["datasets"]["train"]['batch_size'])
        squeduler.step()
        counter += 1
        if epoch % config.get("save_every", 3) == 0:
            torch.save(model.state_dict(), f'{config["weights_dir"]}/{epoch}.pth')    
        if res_v["meteor"] > best_val_metric:
            counter = 0
            best_val_loss = res_v["meteor"]
            print("New best validation meteor")
            torch.save(model.state_dict(), f'{config["root_dir"]}/best_meteor.pth') 
        if counter >= config["patience"]:
            break

    # weights = os.listdir(config["weights_dir"])
    # epoch_best = max([int(i.split('.')[0]) for i in weights])
    
    #epoch_best = weights[:-1].split('.')[0] # Tambe aixi val

    model.load_state_dict(torch.load(f'{config["weights_dir"]}/best_meteor.pth'))
    model.eval()
    loss_t, res_t = eval_epoch(model, crit, metric, dataloader_test, epoch="test")
    wandb.log({"Test Loss": loss_t, "Test": res_t}, step=(epoch+1)*len(dataloader_train)*config["datasets"]["train"]['batch_size'])
    print(f'Test loss: {loss_t:.2f}')
    print(f'Test: {res_t}')

    wandb.finish()