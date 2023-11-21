import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
from torchvision.utils import save_image
import wandb
from tqdm import tqdm
import evaluate
import pandas as pd
import numpy as np
import os
from torchvision.transforms import v2


from data_utils.dataset import Data_char
from data_utils.utils import LoadConfig, metrics_evaluation, get_scheduler
from Models.baseline_tf import Baseline

chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

def train_one_epoch(model, optimizer, crit, metric, dataloader):
    loss = 0
    model.train()

    list_pred = []
    list_ref = []
    for i, (img, cap_idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # load it to the active device
        img = img.to(device)
        cap_idx = cap_idx.to(device) # batch, seq
        outputs = model(img, cap_idx) # batch, 80, seq
        outputs = outputs.argmax(dim=1) # batch, seq

        # Reshape cap_idx to be 1D
        cap_idx = cap_idx.view(-1)

        train_loss = crit(outputs.float(), cap_idx.float())  # Convert cap_idx to float32
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        # wandb.log({"Train Loss": train_loss.item()})
        # Log loss to wandb every 3 batches
        if i % 3 == 0:
            wandb.log({"Train Loss": train_loss.item()}, step = len(dataloader)*len(img)*(epoch)+i*len(img))
        # prob, preds = torch.max(outputs, dim=1)
        # preds = preds.permute(1, 0)
        preds = outputs
        preds = preds.cpu().numpy()
        cap_idx = cap_idx.cpu().numpy()
        for i in range(len(preds)):
            pred = preds[i]
            ref = cap_idx[i]
            pred = [idx2char[j] for j in pred if j not in (0, 1, 2)]
            ref = [idx2char[j] for j in ref if j not in (0, 1, 2)]
            pred = ''.join(pred)
            ref = ''.join(ref)
            list_pred.append(pred)
            list_ref.append(ref)
    res = metrics_evaluation(list_pred, list_ref)

    return loss/len(dataloader), res

def eval_epoch(model, crit, metric, dataloader):
    model.eval()
    loss = 0
    list_pred = []
    list_ref = []
    with torch.no_grad():
        for img, cap_idx in tqdm(dataloader):
            img = img.to(device)
            cap_idx = cap_idx.to(device)
            outputs = model(img, cap_idx)
            outputs = outputs.argmax(dim=1) # batch, seq

            val_loss = crit(outputs, cap_idx.float())  # Convert cap_idx to float32
            loss += val_loss.item()

            # prob, preds = torch.max(outputs, dim=1)
            # preds = preds.permute(1, 0)
            preds = outputs
            preds = preds.cpu().numpy()
            cap_idx = cap_idx.cpu().numpy()
            for i in range(len(preds)):
                pred = preds[i]
                ref = cap_idx[i]
                pred = [idx2char[j] for j in pred if j not in (0, 1, 2)]
                ref = [idx2char[j] for j in ref if j not in (0, 1, 2)]
                pred = ''.join(pred)
                ref = ''.join(ref)
                list_pred.append(pred)
                list_ref.append(ref)
    res = metrics_evaluation(list_pred, list_ref)

    # Save the last 5 images and the predicted captions
    os.makedirs(f"{config['output_dir']}Epoch_{epoch}", exist_ok=True)
    # Trransform to do an inverse normalitzation of the image
    transform = v2.Compose([v2.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                        std=[1/0.229, 1/0.224, 1/0.255])])
    for i in range(5):
        image = img[-i].cpu()
        # De normalize the image
        image = transform(image)
        save_image(image, f'{config["output_dir"]}Epoch_{epoch}/Caption_{list_pred[-i]}_{i}.png')
    return loss/len(dataloader), res




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='run1')
    args = parser.parse_args()
    config = LoadConfig(args.test_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
    img_path = f'{base_path}Images/'
    cap_path = f'{base_path}captions.txt'
    path_partitions = f'{base_path}flickr8k_partitions.npy'
    data = pd.read_csv(cap_path)

    best_val_loss = float('inf')
    TEXT_MAX_LEN = 201

    partitions = np.load(path_partitions, allow_pickle=True).item()
    partitions['train'] = partitions['train'][:int(len(partitions['train'])*0.01)]
    partitions['valid'] = partitions['valid'][:int(len(partitions['valid'])*0.01)]
    partitions['test'] = partitions['test'][:int(len(partitions['test'])*0.01)]
    # config = {"lr":0.001, "weights_dir":"/fhome/gia07/Image_captioning/Test_weights", 
    #           "epochs":10, "datasets":{"train":{"batch_size":32, "shuffle":True, "num_workers":4}}
    #         }
    metric = metrics_evaluation
    with wandb.init(project='test', config=config, name=args.test_name) as run:
        dataset_train = Data_char(data, partitions['train'])
        # Get only a 1% of the dataset to test if the model is able to overfit
        # dataset_train, _ = torch.utils.data.random_split(dataset_train, [int(len(dataset_train)*0.005), len(dataset_train)-int(len(dataset_train)*0.005)])
        dataloader_train = torch.utils.data.DataLoader(dataset_train, **config["datasets"]["train"])
        dataset_valid = Data_char(data, partitions['valid'])
        # Get only a 1% of the dataset to test if the model is able to overfit
        # dataset_valid, _ = torch.utils.data.random_split(dataset_valid, [int(len(dataset_valid)*0.01), len(dataset_valid)-int(len(dataset_valid)*0.01)])
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, **config["datasets"]["valid"])
        dataset_test = Data_char(data, partitions['test'])
        dataloader_test = torch.utils.data.DataLoader(dataset_test, **config["datasets"]["test"])
        model = Baseline(TEXT_MAX_LEN, device).to(device)
        wandb.watch(model)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        crit = nn.CrossEntropyLoss()
        metric = "metrics_evaluation"
        counter = 0
        squeduler = get_scheduler(config, optimizer)
        for epoch in range(config["epochs"]):
            loss, res = train_one_epoch(model, optimizer, crit, metric, dataloader_train)
            print(f'train loss: {loss:.2f}, epoch: {epoch}')
            loss_v, res_v = eval_epoch(model, crit, metric, dataloader_valid)
            print(f'valid loss: {loss:.2f}')
            wandb.log({"Epoch Train Loss": loss, "Epoch Validation Loss": loss_v, "epoch":epoch+1}, step=(epoch+1)*len(dataloader_train)*config["datasets"]["train"]['batch_size'])
            wandb.log({"Train": res, "Validation": res_v, "epoch":epoch+1, "lr":squeduler.get_lr}, step=(epoch+1)*len(dataloader_train)*config["datasets"]["train"]['batch_size'])
            squeduler.step(loss_v)
            counter += 1
            if loss_v < best_val_loss:
                counter = 0
                best_val_loss = loss_v
                print("New best validation loss")
                torch.save(model.state_dict(), f'{config["weights_dir"]}/{epoch}.pth')    
            if counter >= config["patience"]:
                break
    
        weights = os.listdir(config["weights_dir"])
        epoch_best = max([int(i.split('.')[0]) for i in weights])
        
        #epoch_best = weights[:-1].split('.')[0] # Tambe aixi val

        model.load_state_dict(torch.load(f'{config["weights_dir"]}/{epoch_best}.pth'))
        loss_t, res_t = eval_epoch(model, crit, metric, dataloader_test)
        wandb.log({"Test Loss": loss_t, "Test": res_t}, step=(epoch+1)*len(dataloader_train)*config["datasets"]["train"]['batch_size'])
        print(f'Test loss: {loss_t:.2f}')
        print(f'Test: {res_t}')