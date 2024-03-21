import os
import sys
import torch
import pandas as pd
import numpy as np
from src.data_utils import create_dataset

base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
img_path = f'{base_path}Images/'
cap_path = f'{base_path}captions.txt'
path_partitions = f'{base_path}flickr8k_partitions.npy'
data = pd.read_csv(cap_path)
NUM_CAPTIONS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

partitions = np.load(path_partitions, allow_pickle=True).item()
TOTAL_MAX_WORDS = 38
# Traintype: Data_word_regions                 # Data_char, Data_word

batch_size = 32

dataset_train = create_dataset(data, partitions['train'], TOTAL_MAX_WORDS, word2idx = word2idx, train=True, augment=True, dataset_name="TrainSet", dataset_type="Data_word_1cap2", device = device)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 0)

dataset_valid = create_dataset(data, partitions['valid'], TOTAL_MAX_WORDS, word2idx = word2idx, train=False, augment=False, dataset_name="ValidSet", dataset_type="Data_word_1cap2", device = device)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size = batch_size, shuffle = False, num_workers = 0)

dataset_test = create_dataset(data, partitions['test'], 40, word2idx = word2idx, train=False, augment=False, dataset_name="TestSet", dataset_type="Data_word_1cap2", device = device)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 0)


from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch.nn as nn
from torchvision.models import convnext_tiny
import pickle
from tqdm import tqdm

fasterrcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
# Freeze the weights
for param in fasterrcnn.parameters():
    param.requires_grad = False
fasterrcnn.eval()

convnext_tiny_model = convnext_tiny(weights = 'IMAGENET1K_V1').to(device)
# Freeze the weights
for param in convnext_tiny_model.parameters():
    param.requires_grad = False
convnext_tiny_model.eval()

num_boxes = 15

features_dic = {}
print("Start loop feature extraction") 
for loader in [dataloader_train, dataloader_valid, dataloader_test]:
    print("New loader")
    for h, (img, cap, img_name) in enumerate(tqdm(loader)):
        img = img.to(device)
        batch_size = img.shape[0]
        # Resize the image to 224x224
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        feat = convnext_tiny_model.features(img) 
        feat = convnext_tiny_model.avgpool(feat) 
        feat = feat.view(batch_size, 768, -1) 
        feat = feat.squeeze(2)
        regions = fasterrcnn(img)

        # for each image in the batch, get the 5 regions with the highest score
        for i in range(len(regions)):
            print(len(regions[i]["labels"]))
            regions[i]["labels"] = regions[i]["labels"][:num_boxes]
            regions[i]["boxes"] = regions[i]["boxes"][:num_boxes]
            regions[i]["scores"] = regions[i]["scores"][:num_boxes]
            
            if len(regions[i]['boxes']) < num_boxes:
                gap = num_boxes - len(regions[i]['boxes'])
                regions[i]['boxes'] = torch.cat((regions[i]['boxes'], (torch.zeros((gap, 4)).to(device))))
                regions[i]["labels"] = torch.cat((regions[i]["labels"], (torch.ones((gap,))*-1).to(device)))
                regions[i]["scores"] = torch.cat((regions[i]["scores"], (torch.ones((gap,))*-1).to(device)))
            

        
        # print(regions)
        
        for j in range(batch_size):
            feat_regions = torch.zeros((1+num_boxes, 768)).to(device)        
            feat_regions[0] = feat[j, :]
            
            # print(regions["labels"][j])
            for i in range(num_boxes):
                one_img = img[j]
                x1, y1, x2, y2 = regions[j]["boxes"][i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                area = (x2-x1)*(y2-y1)
                if area < 1000:
                    continue
                img_aux = one_img[:, y1:y2, x1:x2]
                # Resize the image to 224x224
                img_aux = torch.nn.functional.interpolate(img_aux.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                
                feat_aux = convnext_tiny_model.features(img_aux) 
                feat_aux = convnext_tiny_model.avgpool(feat_aux) 
                feat_aux = feat_aux.view(1, 768, -1) 
                feat_regions[i+1] = feat_aux.squeeze(2)
            
            # feat_regions = feat_regions.permute(0, 1, 2) # batch, num_regions, out_cnn
            
            features_dic[img_name[j]] = feat_regions.cpu().detach().numpy()

        torch.cuda.empty_cache()
        del feat
        del regions
        del feat_regions

        if h%100 == 0:
            pickle.dump(features_dic, open(f'features_new.pkl', 'wb'))
        # break

pickle.dump(features_dic, open(f'features_new.pkl', 'wb'))
    
os.rename(f'features_new.pkl', f'/fhome/gia07/Image_captioning/src/Dataset/features_new.pkl')
    
features = pickle.load(open(f'/fhome/gia07/Image_captioning/src/Dataset/features_new.pkl', 'rb'))
print(len(features.keys()))

# correct_dict = {}
# for key, value in features.items():
#     print(key, value)
    
#     break
#         # correct_dict[name] = value[idx]
        
    # break
    
    
    