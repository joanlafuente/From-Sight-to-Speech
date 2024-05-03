import os
import sys
import torch
import pandas as pd
import numpy as np
from src.data_utils import create_dataset
from PIL import Image
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
import torchvision
from torchvision.transforms import v2
from torchvision.models.detection.image_list import ImageList
import torch.nn as nn

# Add progres bar
from tqdm import tqdm

base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
img_path = f'{base_path}Images/'
cap_path = f'{base_path}captions.txt'
path_partitions = f'{base_path}flickr8k_partitions.npy'
data = pd.read_csv(cap_path)
NUM_CAPTIONS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

partitions = np.load(path_partitions, allow_pickle=True).item()


# OUT DIR
save_dir = '/fhome/gia07/Image_captioning/src/Dataset/roi_features_fastercnn_coco/'

# Faster R-CNN

import torchvision.models as models
import torch

class CustomFasterRCNN(torch.nn.Module):
    def __init__(self):
        super(CustomFasterRCNN, self).__init__()
        # Load a pre-trained Faster R-CNN model
        #self.fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.fasterrcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)

        self.fasterrcnn.eval()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.fasterrcnn = nn.DataParallel(self.fasterrcnn)
            self.fasterrcnn = self.fasterrcnn.to(device).module
    
    @torch.no_grad()
    def forward(self, images):
        # Get the features from the backbone
        features = self.fasterrcnn.backbone(images)

        images = ImageList(images, [img.shape[-2:] for img in images])

        proposals, _ = self.fasterrcnn.rpn(images, features)
        
        rows = [proposal.shape[0] for proposal in proposals]
        
        box_features = self.fasterrcnn.roi_heads.box_roi_pool(features, proposals, images.image_sizes)

        box_features = self.fasterrcnn.roi_heads.box_head(box_features)
        
        return box_features, rows

class LoaderIMG(Dataset):
    def __init__(self, partition_name):
        self.img_idx = partitions[partition_name]
        self.IMGS = data.iloc[self.img_idx]['image'].tolist()
        self.to_img = v2.ToImage()
        size = 300
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size), antialias=True))
        
    def __len__(self):
        return len(self.IMGS)
    
    def __getitem__(self, idx):
        img_name = self.IMGS[idx]
        img = Image.open(f'{img_path}{img_name}').convert('RGB')
        img = self.img_proc_val(img)
        return img
        
# Example usage
model = CustomFasterRCNN() # Switch to evaluation mode

train_dataset = LoaderIMG(partition_name = 'train')
val_dataset = LoaderIMG(partition_name = 'valid')
test_dataset = LoaderIMG(partition_name = 'test')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)
tes_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)


model = model.to(device)

print("Starting Training Loop...")
features_imgs_train = []

for imgs in tqdm(train_loader):
    imgs = imgs.to(device)
    out, rows = model(imgs)
    start = 0
    for i in range(len(rows)):
        end = start + rows[i]
        features_imgs_train.append(out[start:end, :])
        start = end
        
    torch.cuda.empty_cache()
    del imgs
    del out
    del rows
# Free memory from GPU

    
# Save features
torch.save(features_imgs_train, save_dir + 'features_imgs_train.pt')
del features_imgs_train

print("Starting Validation Loop...")
features_imgs_val = []

for imgs in tqdm(val_loader):
    imgs = imgs.to(device)
    out, rows = model(imgs)
    start = 0
    for i in range(len(rows)):
        end = start + rows[i]
        features_imgs_val.append(out[start:end, :])
        start = end
    torch.cuda.empty_cache()
    del imgs
    del out
    del rows
    
# Save features
torch.save(features_imgs_val, save_dir + 'features_imgs_val.pt')

print("Starting Test Loop...")
features_imgs_test = []

for imgs in tqdm(tes_loader):
    imgs = imgs.to(device)
    out, rows = model(imgs)
    start = 0
    for i in range(len(rows)):
        end = start + rows[i]
        features_imgs_test.append(out[start:end, :])
        start = end
    torch.cuda.empty_cache()
    del imgs
    del out
    del rows
# Save features
torch.save(features_imgs_test, save_dir + 'features_imgs_test.pt')

print("Done!")