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
import json

from Models import define_network # esta al init de models
from data_utils import create_dataset # esta al init de data_utils
from data_utils.utils import *
from data_utils.dataset import *

CUDA_LAUNCH_BLOCKING = 1
# TORCH_USE_CUDA_DSA = 1


def eval_epoch(model, crit, metric, dataloader, epoch=0, print_attention = False):
    # model.eval()
    loss = 0
    list_pred = []
    list_ref = []
    with torch.no_grad():
        for i, (img, cap_idx) in enumerate(tqdm(dataloader)):
            img = img.to(device)
            cap_idx = cap_idx.to(device)
            # regions = regions.to(device)
            outputs = model(img)

            temp_val_loss = torch.tensor([]).to(device)
            for j in range(5):
                val_loss = crit(outputs, cap_idx[:, j, 1:])
                temp_val_loss = torch.cat((temp_val_loss, val_loss.unsqueeze(0)), dim=0)
            
            val_loss = crit(outputs, cap_idx[:, temp_val_loss.argmin(dim=0), 1:])
            # val_loss = val_loss.min(dim=1)[0]
            loss += val_loss.item()

            _, preds = torch.max(outputs, dim=1)
            # preds = preds.permute(1, 0)
            preds = preds.cpu().numpy()
            cap_idx = cap_idx.cpu().numpy()
        
            if i < 5:
                print()
                print("-------------------------------------------------------------------------") 
                print("Loss", val_loss.item())
                print(' '.join([idx2word[word] for word in preds[0]]))
        
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
    if (epoch != "test") and (epoch > config["network"]["epoch2unfreeze"]):
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
    
    # if print_attention:
    #     print(save_output.outputs[0][0, 0, :, :])   
             
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

    with open(f'{config["root_dir"]}word2idx.json', 'r') as f:
        word2idx = json.load(f)

    idx2word = {v: k for k, v in word2idx.items()}
    NUM_WORDS = len(word2idx)

    partitions = np.load(path_partitions, allow_pickle=True).item()
    TOTAL_MAX_WORDS = config['network']['params']['text_max_len']
    
    num_classes = len(list(word2idx.keys()))
    
    print("using penalize_pad")
    weights = torch.ones(num_classes)
    weights[word2idx['<PAD>']] = 0.1
    
    
    # Train
    dataset_train = create_dataset(data, partitions['train'], TOTAL_MAX_WORDS, word2idx = word2idx, train=True, augment=config['datasets']['augment_imgs'], dataset_name="TrainSet", dataset_type=config["datasets"]["type"])# Data_word_aug(data, partitions['train'], train=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, **config["datasets"]["train"])
    # Val
    dataset_valid = create_dataset(data, partitions['valid'], TOTAL_MAX_WORDS, word2idx = word2idx, train=False, augment=False, dataset_name="ValidSet", dataset_type=config["datasets"]["type"])# Data_word_aug(data, partitions['val'], train=False)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, **config["datasets"]["valid"])
    # Test
    dataset_test = create_dataset(data, partitions['test'], TOTAL_MAX_WORDS, word2idx = word2idx, train=False, augment=False, dataset_name="TestSet", dataset_type=config["datasets"]["type"])# Data_word_aug(data, partitions['test'], train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **config["datasets"]["test"])
    
    config["network"]["params"]["NUM_WORDS"] = NUM_WORDS
    config["network"]["params"]["word2idx"]  = word2idx
    config["network"]["params"]["idx2word"]  = idx2word

    
    # print("NUM_WORDS", NUM_WORDS)

    model = define_network(config["network"]["params"]) # LSTM_attention(TOTAL_MAX_WORDS, **config["network"]["params"], device = device).to(device)

    # save_output = SaveOutput()
    # patch_attention(model.transformer_decoder.layers[-1].self_attn)
    # hook_handle = model.transformer_decoder.layers[-1].self_attn.register_forward_hook(save_output)

    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    print(torch.cuda.device_count())
    
    model = model.to(device)
    
    crit = get_loss(config["loss"], weights = weights, device = device)
    metric = metrics_evaluation
    metric_earlystop = config["early_stopping"]["metric"]

    wandb.init(project='Image_Captioning2', config=config, name=args.test_name)
    wandb.watch(model)

    last_epoch = os.listdir(config["weights_dir"])
    last_epoch = max([int(epoch.split(".")[0]) for epoch in last_epoch])
    model.load_state_dict(torch.load(f'{config["weights_dir"]}{last_epoch}.pth'))
    
    loss_t, res_t = eval_epoch(model, crit, metric, dataloader_test, epoch="test")
    wandb.log({"Test Loss": loss_t, "Test": res_t})
    with open(f"{config['root_dir']}test_results_epoch_{last_epoch}.json", "w") as f:
        json.dump(res_t, f)
    print(f'Test loss: {loss_t:.2f}')
    print(f'Test: {res_t}')

    model.load_state_dict(torch.load(f'{config["root_dir"]}best_{metric_earlystop}.pth'))

    loss_t, res_t = eval_epoch(model, crit, metric, dataloader_test, epoch="test")
    wandb.log({"Test Loss": loss_t, "Test": res_t})

    with open(f"{config['root_dir']}test_results_best_{metric_earlystop}.json", "w") as f:
        json.dump(res_t, f)
    print(f'Test loss: {loss_t:.2f}')
    print(f'Test: {res_t}')

    wandb.finish()
    
    
    
    # ['<SOS> a a boy <PAD> his <PAD> <PAD> <PAD> <PAD> his <PAD> <PAD> head <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> head <PAD> <PAD> <PAD> head <PAD> <PAD> hands <PAD> <PAD> his <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>', 
    # '<SOS> a a a a leather a <PAD> little a <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> a <PAD> a <PAD> a <PAD> <PAD> a <PAD> little <PAD> <PAD> a <PAD> <PAD> a <PAD> <PAD> a little <PAD> <PAD> <PAD>', 
    # '<SOS> a baby baby over <PAD> in <PAD> <PAD> baby <PAD> with <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> with <PAD> food <PAD> baby food <PAD> <PAD> with <PAD> <PAD> <PAD> <PAD> <PAD> a', 
    # '<SOS> climbers two climbers <PAD> <PAD> climbers <PAD> are <PAD> are <PAD> are <PAD> <PAD> <PAD> <PAD> top <PAD> <PAD> climbers <PAD> are <PAD> <PAD> <PAD> climbers <PAD> sitting <PAD> <PAD> <PAD> sitting top <PAD> <PAD> <PAD> top <PAD> <PAD>', '<SOS> coming a race race race <PAD> around <PAD> <PAD> <PAD> <PAD> <PAD> coming <PAD> a <PAD> <PAD> <PAD> <PAD> coming race <PAD> <PAD> <PAD> <PAD> dust <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> race <PAD> <PAD> <PAD> <PAD> <PAD>', '<SOS> being is boxer being <PAD> examined his <PAD> <PAD> <PAD> examined <PAD> <PAD> <PAD> boxer <PAD> being being <PAD> being <PAD> <PAD> <PAD> <PAD> boxer <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> examined <PAD> being <PAD> <PAD> his <PAD> <PAD>', '<SOS> a man onstage beside going fireworks <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> with <PAD> fireworks <PAD> <PAD> <PAD> <PAD> is <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> fireworks <PAD> <PAD> man <PAD> <PAD> onstage <PAD> <PAD> fireworks', '<SOS> a a being shop is perused <PAD> shop <PAD> shop <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> being <PAD> shop <PAD> <PAD> sitting <PAD> <PAD> <PAD> garden <PAD> <PAD> <PAD> <PAD> shop <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>', '<SOS> a black black <PAD> dog bird <PAD> <PAD> bird bird <PAD> <PAD> <PAD> a <PAD> <PAD> jumps <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> a', '<SOS> an <SOS> <PAD> <PAD> <PAD> elderly <PAD> <PAD> orange <PAD> orange <PAD> <PAD> <PAD> orange <PAD> <PAD> orange <PAD> <PAD> elderly <PAD> <PAD> <PAD> orange <PAD> orange <PAD> <PAD> <PAD> overalls <PAD> <PAD> <PAD> <PAD> overalls <PAD> <PAD> <PAD>', '<SOS> a a <PAD> a <PAD> dog <PAD> a <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> and <PAD> <PAD> and <PAD> <PAD> biting <PAD> and <PAD> <PAD> <PAD> <PAD> <PAD> biting <PAD> <PAD> biting <PAD> and <PAD> <PAD> biting', '<SOS> a orange <PAD> girl <PAD> girl <PAD> girl <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> girl <PAD> <PAD> shirt <PAD> <PAD> girl <PAD> <PAD> <PAD> <PAD> girl <PAD> girl <PAD> <PAD> <PAD> <PAD> a <PAD> girl <PAD> <PAD> <PAD> girl', '<SOS> a boy boy boy <PAD> boy boy <PAD> at <PAD> <PAD> <PAD> <PAD> boy <PAD> boy <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> red <PAD> <PAD> <PAD> <PAD> boy <PAD> boy <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>', '<SOS> one dog dog dog dog <PAD> dog <PAD> <PAD> dog up <PAD> <PAD> <PAD> <PAD> dog <PAD> <PAD> <PAD> <PAD> <PAD> one <PAD> dog <PAD> dog <PAD> <PAD> jumping <PAD> dog <PAD> <PAD> <PAD> <PAD> <PAD> one <PAD> <PAD>', '<SOS> a <PAD> running running brown <PAD> dog <PAD> through running <PAD> <PAD> is <PAD> running <PAD> is <PAD> <PAD> running <PAD> running running <PAD> <PAD> <PAD> <PAD> is <PAD> <PAD> <PAD> <PAD> running <PAD> <PAD> <PAD> <PAD> <PAD> a', '<SOS> a a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> a <PAD> boy <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> boy <PAD>', '<SOS> <SOS> curly a <PAD> curly is curly <PAD> <PAD> <PAD> <PAD> is <PAD> <PAD> <PAD> into <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> haired <PAD> is <PAD> <PAD> <PAD> dragging <PAD> <PAD> is <PAD> a <PAD> <PAD> dragging <PAD>', '<SOS> blonde blonde ladies on two <PAD> ladies on <PAD> on <PAD> lounge <PAD> wearing <PAD> <PAD> wearing <PAD> <PAD> <PAD> on <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> wearing <PAD> <PAD> <PAD> <PAD> wearing <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>', '<SOS> a retriever <PAD> playing and golden <PAD> and <PAD> <PAD> <PAD> and <PAD> <PAD> <PAD> retriever <PAD> golden <PAD> and <PAD> playing <PAD> <PAD> dog <PAD> <PAD> <PAD> <PAD> dog <PAD> <PAD> <PAD> <PAD> retriever <PAD> <PAD> <PAD> golden', '<SOS> a <SOS> couple couple <PAD> <PAD> <PAD> hug <PAD> <PAD> <PAD> <PAD> each hug <PAD> hug <PAD> <PAD> hug <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> hug <PAD> <PAD> tiger <PAD> <PAD> couple <PAD> each <PAD> each <PAD> <PAD>', '<SOS> a boy boy pink <PAD> a boy <PAD> <PAD> pink <PAD> <PAD> red <PAD> <PAD> pink <PAD> <PAD> a <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD> and <PAD> <PAD> <PAD> girl <PAD> pink <PAD> and <PAD> <PAD> <PAD> <PAD>', '<SOS> a a on a a <PAD> <PAD> white <PAD> a <PAD> a <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> top <PAD> <PAD> a <PAD> <PAD> a <PAD> <PAD> <PAD> baby <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD>', '<SOS> a and a <PAD> dog <PAD> and <PAD> white <PAD> jumping <PAD> <PAD> dog <PAD> jumping <PAD> <PAD> dog <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> dog <PAD> white <PAD> <PAD> jumping', '<SOS> there a being <PAD> being is <PAD> <PAD> <PAD> <PAD> <PAD> is <PAD> <PAD> <PAD> is <PAD> is <PAD> <PAD> college <PAD> <PAD> <PAD> <PAD> <PAD> is <PAD> is <PAD> <PAD> player <PAD> <PAD> <PAD> is <PAD> is <PAD>', '<SOS> a white boy <PAD> a <PAD> <PAD> boy <PAD> boy <PAD> boy <PAD> <PAD> <PAD> boy <PAD> boy <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> boy <PAD> boy <PAD> <PAD> <PAD> <PAD> a <PAD> baseball <PAD> <PAD>']
    # ['<SOS> a a <PAD> on <PAD> laying <PAD> <PAD> <PAD> <PAD> on <PAD> <PAD> on <PAD> <PAD> is <PAD> <PAD> laying <PAD> a <PAD> reaching <PAD> <PAD> on <PAD> red <PAD> <PAD> <PAD> on <PAD> a <PAD> a <PAD> shirt', '<SOS> a is <PAD> <PAD> <PAD> shorts <PAD> <PAD> shorts <PAD> <PAD> <PAD> boy <PAD> <PAD> <PAD> <PAD> <PAD> under <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> is <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> back <PAD> doing <PAD> <PAD> boy <PAD>', '<SOS> pack of pack <PAD> <PAD> <PAD> dogs <PAD> dogs <PAD> <PAD> <PAD> <PAD> <PAD> running <PAD> black <PAD> <PAD> <PAD> <PAD> dogs <PAD> <PAD> <SOS> <PAD> a <PAD> dogs <PAD> <PAD> <PAD> pack <PAD> of <PAD> <PAD> <PAD> <PAD>', '<SOS> a himself a wearing <PAD> <PAD> <PAD> himself <PAD> <PAD> <PAD> <PAD> himself <PAD> <PAD> a <PAD> himself <PAD> <PAD> <PAD> <PAD> pulling <PAD> up <PAD> a <PAD> headband <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> himself <PAD> <PAD> <PAD>', 'a boy boy basketball <PAD> boy <PAD> <PAD> basketball <PAD> basketball <PAD> boy <PAD> <PAD> <PAD> <PAD> <PAD> holding <PAD> <PAD> boy <PAD> basketball <PAD> basketball <PAD> <PAD> basketball <PAD> <PAD> <PAD> basketball <PAD> a <PAD> <PAD> basketball <PAD> <PAD>', '<SOS> are people <PAD> <PAD> people <PAD> <PAD> <PAD> <PAD> are <PAD> <PAD> <PAD> the <PAD> are <PAD> <PAD> <PAD> <PAD> are <PAD> are <PAD> are <PAD> on <PAD> <PAD> <PAD> step <PAD> are <PAD> on <PAD> <PAD> <PAD> <PAD>', '<SOS> falling girl <PAD> <PAD> a <PAD> <PAD> <PAD> in <PAD> in <PAD> falling <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> falling <PAD> <PAD> falling <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD>', '<SOS> child <PAD> <PAD> child <PAD> father <PAD> feeding <PAD> a <PAD> <PAD> father <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> child <PAD> father <PAD> feeding', '<SOS> dog <PAD> dog dog dog <PAD> dog <PAD> dog dog <PAD> a <PAD> <PAD> dog <PAD> <PAD> <PAD> <PAD> up <PAD> <PAD> up <PAD> <PAD> dog <PAD> dog <PAD> up <PAD> <PAD> <PAD> dog <PAD> dog <PAD> jumping <PAD>', '<SOS> group around around <PAD> <PAD> <PAD> <PAD> of <PAD> a <PAD> group <PAD> <PAD> <PAD> <PAD> <PAD> dogs <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> dogs <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>', '<SOS> a a boy <PAD> in <PAD> a <PAD> <PAD> <PAD> the in <PAD> a <PAD> a <PAD> <PAD> pumpkin <PAD> boy <PAD> in <PAD> pumpkin <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> in <PAD> <PAD> <PAD> <PAD>', '<SOS> little black little <PAD> chases <PAD> <PAD> a <PAD> little <PAD> dog <PAD> <PAD> chases <PAD> chases <PAD> <PAD> <PAD> <PAD> <PAD> chases <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> little <PAD> <PAD> <PAD> <PAD>', '<SOS> a brown dog <PAD> <PAD> <PAD> <PAD> <PAD> dog <PAD> <PAD> <PAD> <PAD> <PAD> dog <PAD> <PAD> <PAD> dog <PAD> brown <PAD> <PAD> <PAD> a <PAD> brown <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>', '<SOS> adults adults winter a <PAD> looks brooms <PAD> brooms <PAD> winter <PAD> looks <PAD> jackets <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> looks <PAD> brooms <PAD> <PAD> brooms <PAD> ball <PAD> ball <PAD> <PAD> <PAD> <PAD> ball <PAD> <PAD> a', 'a dog dog <PAD> <PAD> <PAD> a <PAD> a <PAD> dog <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> dog <PAD> a <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> dog <PAD>', '<SOS> a <PAD> is <PAD> <PAD> <PAD> <PAD> boy <PAD> is <PAD> girl <PAD> a <PAD> boy <PAD> <PAD> <PAD> a <PAD> <PAD> little <PAD> <PAD> boy <PAD> <PAD> <PAD> <PAD> <PAD> boy <PAD> <PAD> girl <PAD> <PAD> <PAD> peaking', '<SOS> group group people <PAD> people <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> people <PAD> <PAD> <PAD> <PAD> <PAD> people <PAD> <PAD> and <PAD> people <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> group <PAD> people <PAD> <PAD> group <PAD>', '<SOS> at at a at <PAD> <PAD> a <PAD> <PAD> at <PAD> <PAD> <PAD> at <PAD> at <PAD> <PAD> <PAD> at <PAD> <PAD> child <PAD> at <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> at <PAD>', '<SOS> <PAD> boy <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> boy <PAD> a <PAD> <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>', '<SOS> pug pug <PAD> up up <PAD> <PAD> <PAD> looking <PAD> up <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> up <PAD> <PAD> dog <PAD> <PAD> <PAD> pug <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>', '<SOS> a <PAD> <PAD> playing playing <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> boy <PAD> playing <PAD> <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> playing <PAD> playing <PAD> a <PAD> <PAD> <PAD> <PAD>', 'a a a buggy <PAD> is <PAD> <PAD> a <PAD> down <PAD> <PAD> a <PAD> is <PAD> <PAD> boy <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> buggy <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>', '<SOS> is a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> air <PAD> <PAD> a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> doing <PAD> a <PAD> a <PAD> doing <PAD> <PAD> <PAD>', '<SOS> in <PAD> begins begins <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> begins <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> begins <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> a <PAD> begins <PAD> <PAD> <PAD>', '<SOS> a a <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> a <PAD> throws <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>']
