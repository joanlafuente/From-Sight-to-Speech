import pandas as pd
base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
cap_path = f'{base_path}captions.txt'

data = pd.read_csv(cap_path)


print(max(data.caption.apply(lambda x: x.split()).values, key=len).__len__())
exit(0)
unique_words = set()
captions = data.caption.apply(lambda x: x.lower()).values

#for i in range(len(data)):
#    caption = captions[i]
#    caption = caption.split()
#    print(caption)
#
#    unique_words.update(caption)
#
#print(unique_words)

import numpy as np
from src.data_utils.dataset import Data_word

base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
img_path = f'{base_path}Images/'
cap_path = f'{base_path}captions.txt'
path_partitions = f'{base_path}flickr8k_partitions.npy'


partitions = np.load(path_partitions, allow_pickle=True).item()



dataset_train = Data_word(data, partitions['train'])

img, cap = dataset_train[0]
print(cap.__len__())