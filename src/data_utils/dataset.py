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

# partitions = np.load(path_partitions, allow_pickle=True).item()
# # chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# chars = ['<SOS>', '<EOS>', '<PAD>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# NUM_CHAR = len(chars)
# idx2char = {k: v for k, v in enumerate(chars)}
# char2idx = {v: k for k, v in enumerate(chars)}
# TEXT_MAX_LEN = 201
# DEVICE = 'cuda'

# # Unique words in the dataset
# unique_words = set()
# captions = data.caption.apply(lambda x: x.lower()).values
# for i in range(len(data)):
#     caption = captions[i]
#     caption = caption.split()
#     unique_words.update(caption)

# NUM_WORDS = len(unique_words)
# unique_words = ['<SOS>', '<EOS>', '<PAD>'] + sorted(list(unique_words))
# idx2word = {k: v for k, v in enumerate(unique_words)}
# word2idx = {v: k for k, v in enumerate(unique_words)}
# TOTAL_MAX_WORDS = 38

# Data set used for the model
class Data_char(Dataset):
    #                  data, partitions, TOTAL_MAX_WORDS, word2idx, train, augment, dataset_name, dataset_type='Data_word', device = None, size=224, type_partition = None
    def __init__(self, data, partition, TOTAL_MAX_WORDS, word2idx, train = True, augment = False, device = None, size=224,  type_partition = None): # Train for ensure comptaibility with Data_word
        self.data = data
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
        self.partition = partition
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        self.chars = list(word2idx.keys())

        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_train = self.img_proc_train if augment else self.img_proc_val
        self.img_proc = self.img_proc_train if train else self.img_proc_val
        self.train = train
        self.char2idx = word2idx

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
        if self.train:
            caption = item.caption.reset_index(drop=True)[0] 
            # caption = item.caption.reset_index(drop=True)[random.choice(list(range(self.num_captions)))]
            cap_list = list(caption)
            final_list = [self.chars[0]]
            final_list.extend(cap_list)
            final_list.extend([self.chars[1]])
            cap_idx = [self.char2idx[i] for i in final_list if i in self.char2idx]
            gap = self.max_len - len(cap_idx)
            cap_idx.extend([self.char2idx[self.chars[2]]]*gap)
            return img, torch.tensor(cap_idx)
        
        else: # Validation and test -> return all captions
            captions = item.caption.apply(lambda x: x.lower()).reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = list(cap)
                final_list = [self.chars[0]] + cap_list + [self.chars[1]]
                cap_idx.append([self.char2idx[i] for i in final_list if i in self.char2idx])
                gap = self.max_len - len(cap_idx[-1])
                cap_idx[-1].extend([self.char2idx[self.chars[2]]]*gap)
            
            return img, torch.tensor(cap_idx)

class Data_word(Dataset):
    def __init__(self, data, partition, TOTAL_MAX_WORDS, word2idx, train=True, augment=False, device=None, size=224,  type_partition = None):
        self.data = data
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())

        self.partition = partition
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        
        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            # v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            # v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5),
            # v2.RandomPosterize(bits=2),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_train = self.img_proc_train if augment else self.img_proc_val
        self.img_proc = self.img_proc_train if train else self.img_proc_val
        self.train = train
        self.word2idx = word2idx
        print("num words dataset", len(list(word2idx.keys())))
        
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
        if self.train:
            #caption = item.caption.reset_index(drop=True)[0]
            caption = item.caption.reset_index(drop=True)[random.choice(list(range(self.num_captions)))]

            cap_list = caption.split()
            final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
            gap = self.max_len - len(final_list)
            final_list.extend([list(self.word2idx.keys())[2]]*gap)
            cap_idx = [self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list]
            return img, torch.tensor(cap_idx)
    
        else: # Validation and test -> return all captions
            captions = item.caption.reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = cap.split()
                final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
                gap = self.max_len - len(final_list)
                final_list.extend([list(self.word2idx.keys())[2]]*gap)
                cap_idx.append([self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list])
            
            return img, torch.tensor(cap_idx)

class Data_word_1cap(Dataset):
    def __init__(self, data, partition, TOTAL_MAX_WORDS, word2idx, train=True, augment=False, device = None, size=224,  type_partition = None):
        self.data = data
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
        self.partition = partition
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_train = self.img_proc_train if augment else self.img_proc_val
        self.img_proc = self.img_proc_train if train else self.img_proc_val
        self.train = train
        self.word2idx = word2idx

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
        if self.train:
            caption = item.caption.reset_index(drop=True)[0]

            cap_list = caption.split()
            final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
            gap = self.max_len - len(final_list)
            final_list.extend([list(self.word2idx.keys())[2]]*gap)
            cap_idx = [self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list]
            return img, torch.tensor(cap_idx)
    
        else: # Validation and test -> return all captions
            captions = item.caption.reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = cap.split()
                final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
                gap = self.max_len - len(final_list)
                final_list.extend([list(self.word2idx.keys())[2]]*gap)
                cap_idx.append([self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list])
            
            return img, torch.tensor(cap_idx)


class Data_word_aug(Dataset):
    def __init__(self, data, partition, TOTAL_MAX_WORDS, word2idx, train=True):
        self.data = data
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())

        self.partition = partition
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            #v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            #v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            #v2.RandomPosterize(bits=2),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        self.train = train
        self.word2idx = word2idx

        
    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        real_idx = self.num_captions*self.partition[idx]
        item = self.data.iloc[real_idx: real_idx+self.num_captions]
        ## image processing
        img_name = item.image.reset_index(drop=True)[0]
        img = Image.open(f'{img_path}{img_name}').convert('RGB')
        ## caption processing
        if self.train:
            img = self.img_proc_train(img)
            caption = item.caption.reset_index(drop=True)[random.choice(list(range(self.num_captions)))]

            cap_list = caption.split()
            final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
            gap = self.max_len - len(final_list)
            final_list.extend([list(self.word2idx.keys())[2]]*gap)
            cap_idx = [self.word2idx[i] for i in final_list]
            return img, torch.tensor(cap_idx)
    
        else: # Validation and test -> return all captions
            img = self.img_proc_val(img)
            captions = item.caption.reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = cap.split()
                final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
                gap = self.max_len - len(final_list)
                final_list.extend([list(self.word2idx.keys())[2]]*gap)
                cap_idx.append([self.word2idx[i] for i in final_list])
            
            return img, torch.tensor(cap_idx)

# Dataset to perform statistics
class Data_stats(Dataset):
    def __init__(self, data, partition, TEXT_MAX_LEN=201):
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
        #list_of_words = []
        #for word in caption.split():
        #    list_of_words.append(word)
        list_of_words = caption.split()
        return img, cap_idx, list_of_words
    
def get_loader_stats(partition, config):
    dataset = Data_stats(data, partitions[partition])
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size = config[partition]['batch_size'], 
                                         shuffle    = config[partition]['shuffle'])
    return loader



from sentence_transformers import SentenceTransformer
class Data_word_sent_embeds(Dataset):
    def __init__(self, data, partition, train=True, augment=False):
        self.data = data
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())

        self.partition = partition
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_train = self.img_proc_train if augment else self.img_proc_val
        self.img_proc = self.img_proc_train if train else self.img_proc_val
        self.train = train

        self.sentenceEmbedder = SentenceTransformer('all-MiniLM-L6-v2')
        
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
        if self.train:
            caption = item.caption.reset_index(drop=True)[0]

            cap_list = caption.split()
            final_list = [chars[0]] + cap_list + [chars[1]]
            gap = self.max_len - len(final_list)
            final_list.extend([chars[2]]*gap)
            cap_idx = [word2idx[i] for i in final_list]
            return img, torch.tensor(cap_idx)
    
        else: # Validation and test -> return all captions
            captions = item.caption.reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = cap.split()
                final_list = [chars[0]] + cap_list + [chars[1]]
                gap = self.max_len - len(final_list)
                final_list.extend([chars[2]]*gap)
                cap_idx.append([word2idx[i] for i in final_list])
            
            return img, torch.tensor(cap_idx)



class Data_word_visuals(Dataset):
    

    def __init__(self, data, partition, TOTAL_MAX_WORDS, word2idx, train=True, augment=False):
        self.data = data
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())

        self.partition = partition
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            #v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            #v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # v2.RandomPosterize(bits=2),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_train = self.img_proc_train if augment else self.img_proc_val
        self.img_proc = self.img_proc_train if train else self.img_proc_val
        self.train = train
        self.word2idx = word2idx
        
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
        if self.train:
            caption = item.caption.reset_index(drop=True)[0]
            #caption = item.caption.reset_index(drop=True)[random.choice(list(range(self.num_captions)))]

            cap_list = caption.split()
            final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
            gap = self.max_len - len(final_list)
            final_list.extend([list(self.word2idx.keys())[2]]*gap)
            cap_idx = final_list
            return img, torch.tensor(cap_idx)
    
        else: # Validation and test -> return all captions
            captions = item.caption.reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = cap.split()
                final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
                gap = self.max_len - len(final_list)
                final_list.extend([list(self.word2idx.keys())[2]]*gap)
                cap_idx.append(final_list)
            
            return img, cap_idx
        
        
import pickle
class Data_word_regions(Dataset):    
    def __init__(self, data, partition, TOTAL_MAX_WORDS, word2idx, train=True, augment=False, device = None, size = None,  type_partition = None):
        self.data = data
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
        self.partition = partition
        self.regions = pickle.load(open(f'/fhome/gia07/Image_captioning/src/Dataset/features.pkl', 'rb'))
        
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        
        self.img_proc_train = self.img_proc_train if augment else self.img_proc_val
        self.img_proc = self.img_proc_train if train else self.img_proc_val
        self.train = train
        self.word2idx = word2idx
        
        
    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        real_idx = self.num_captions*self.partition[idx]
        item = self.data.iloc[real_idx: real_idx+self.num_captions]
        ## image processing
        img_name = item.image.reset_index(drop=True)[0]
        img_pil = Image.open(f'{img_path}{img_name}') #.convert('RGB')
        img = self.img_proc(img_pil)
        
        regions = self.regions[img_name]
        
        ## caption processing
        if self.train:
            caption = item.caption.reset_index(drop=True)[0]

            cap_list = caption.split()
            final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
            gap = self.max_len - len(final_list)
            final_list.extend([list(self.word2idx.keys())[2]]*gap)
            cap_idx = [self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list]
            return img, torch.tensor(cap_idx), regions
    
        else: # Validation and test -> return all captions
            captions = item.caption.reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = cap.split()
                final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
                gap = self.max_len - len(final_list)
                final_list.extend([list(self.word2idx.keys())[2]]*gap)
                cap_idx.append([self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list])
            
            return img, torch.tensor(cap_idx), regions


class Data_word_1cap2(Dataset):
    def __init__(self, data, partition, TOTAL_MAX_WORDS, word2idx, train=True, augment=False, device = None, size=224,  type_partition = None):
        self.data = data
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
        self.partition = partition
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((size, size), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), )
        
        self.img_proc_train = self.img_proc_train if augment else self.img_proc_val
        self.img_proc = self.img_proc_train if train else self.img_proc_val
        self.train = train
        self.word2idx = word2idx

        
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
        if self.train:
            caption = item.caption.reset_index(drop=True)[0]

            cap_list = caption.split()
            final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
            gap = self.max_len - len(final_list)
            final_list.extend([list(self.word2idx.keys())[2]]*gap)
            cap_idx = [self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list]
            return img, torch.tensor(cap_idx), img_name
    
        else: # Validation and test -> return all captions
            captions = item.caption.reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = cap.split()
                final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
                gap = self.max_len - len(final_list)
                final_list.extend([list(self.word2idx.keys())[2]]*gap)
                cap_idx.append([self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list])
            
            return img, torch.tensor(cap_idx), img_name


class Data_word_regions_rnn_1cap(Dataset):    
    def __init__(self, data, partition, TOTAL_MAX_WORDS, word2idx, train=True, augment=False, device = None, size = None,  type_partition = None):
        self.data = data
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
        self.partition = partition
        self.regions = pickle.load(open(f'/fhome/gia07/Image_captioning/src/Dataset/features.pkl', 'rb'))
        
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        
        self.img_proc_train = self.img_proc_train if augment else self.img_proc_val
        self.img_proc = self.img_proc_train if train else self.img_proc_val
        self.train = train
        self.word2idx = word2idx
        
        
    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        real_idx = self.num_captions*self.partition[idx]
        item = self.data.iloc[real_idx: real_idx+self.num_captions]
        ## image processing
        img_name = item.image.reset_index(drop=True)[0]
        # img_pil = Image.open(f'{img_path}{img_name}') #.convert('RGB')
        # img = self.img_proc(img_pil)
        
        regions = self.regions[img_name]
        
        ## caption processing
        if self.train:
            caption = item.caption.reset_index(drop=True)[0]

            cap_list = caption.split()
            final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
            gap = self.max_len - len(final_list)
            final_list.extend([list(self.word2idx.keys())[2]]*gap)
            cap_idx = [self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list]
            return regions, torch.tensor(cap_idx)
    
        else: # Validation and test -> return all captions
            captions = item.caption.reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = cap.split()
                final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
                gap = self.max_len - len(final_list)
                final_list.extend([list(self.word2idx.keys())[2]]*gap)
                cap_idx.append([self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list])
            
            return regions, torch.tensor(cap_idx)
        
        

class Data_word_regions_rnn_convnext_1cap(Dataset):    
    def __init__(self, data, partition, TOTAL_MAX_WORDS, word2idx, train=True, augment=False, device = None, size = None,  type_partition = None):
        self.data = data
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
        self.partition = partition
        self.regions = pickle.load(open(f'/fhome/gia07/Image_captioning/src/Dataset/features_new.pkl', 'rb'))
        
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        
        self.img_proc_train = self.img_proc_train if augment else self.img_proc_val
        self.img_proc = self.img_proc_train if train else self.img_proc_val
        self.train = train
        self.word2idx = word2idx
        
        
    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        real_idx = self.num_captions*self.partition[idx]
        item = self.data.iloc[real_idx: real_idx+self.num_captions]
        ## image processing
        img_name = item.image.reset_index(drop=True)[0]
        # img_pil = Image.open(f'{img_path}{img_name}') #.convert('RGB')
        # img = self.img_proc(img_pil)
        
        regions = self.regions[img_name]
        
        ## caption processing
        if self.train:
            caption = item.caption.reset_index(drop=True)[0]

            cap_list = caption.split()
            final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
            gap = self.max_len - len(final_list)
            final_list.extend([list(self.word2idx.keys())[2]]*gap)
            cap_idx = [self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list]
            return regions, torch.tensor(cap_idx)
    
        else: # Validation and test -> return all captions
            captions = item.caption.reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = cap.split()
                final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
                gap = self.max_len - len(final_list)
                final_list.extend([list(self.word2idx.keys())[2]]*gap)
                cap_idx.append([self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list])
            
            return regions, torch.tensor(cap_idx)
    
import torch.nn.functional as F
class Data_Word_mobilnet_roifeat(Dataset):    
    def __init__(self, data, partition, TOTAL_MAX_WORDS, word2idx, train=True, augment=False, device = None, size = None, type_partition = None):
        self.data = data # Captions
        self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
        self.partition = partition
        save_dir = '/fhome/gia07/Image_captioning/src/Dataset/roi_features_fastercnn_coco/'
        mobilnet_features_name = {'train': 'mobilenet_train.pt', 'valid': 'mobilenet_val.pt', 'test': 'mobilenet_test.pt'}[type_partition]
        self.mobilnet_features = torch.load(save_dir + mobilnet_features_name, map_location=torch.device('cpu'))
        self.MAX_NUM_REGIONS = 1000
        
        self.num_captions = 5
        self.max_len = TOTAL_MAX_WORDS
        
        self.train = train
        self.word2idx = word2idx
        
        self.img_proc_train = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.4),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        
        self.img_proc_val = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        
        self.img_proc_train = self.img_proc_train if augment else self.img_proc_val
        self.img_proc = self.img_proc_train if train else self.img_proc_val
        
    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx): 
        roi_features = self.mobilnet_features[idx]
        padding = self.MAX_NUM_REGIONS - roi_features.shape[0]
        roi_features = F.pad(roi_features, (0, 0, 0, padding))
        
        real_idx = self.num_captions*self.partition[idx]
        item = self.data.iloc[real_idx: real_idx+self.num_captions]
        
        img_name = item.image.reset_index(drop=True)[0]
        img = Image.open(f'{img_path}{img_name}').convert('RGB')
        img = self.img_proc(img)

        if self.train:
            caption = item.caption.reset_index(drop=True)[0]

            cap_list = caption.split()
            final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
            gap = self.max_len - len(final_list)
            final_list.extend([list(self.word2idx.keys())[2]]*gap)
            cap_idx = [self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list]
            return (roi_features, img), torch.tensor(cap_idx)
    
        else: # Validation and test -> return all captions
            captions = item.caption.reset_index(drop=True)
            cap_idx = []
            for cap in captions:
                cap_list = cap.split()
                final_list = [list(self.word2idx.keys())[0]] + cap_list + [list(self.word2idx.keys())[1]]
                gap = self.max_len - len(final_list)
                final_list.extend([list(self.word2idx.keys())[2]]*gap)
                cap_idx.append([self.word2idx[i] if i in self.word2idx.keys() else self.word2idx['<UNK>'] for i in final_list])
            
            return (roi_features, img), torch.tensor(cap_idx)
    