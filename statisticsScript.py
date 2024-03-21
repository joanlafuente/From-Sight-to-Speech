from tqdm import tqdm
import numpy as np
import pandas as pd

from src.data_utils.dataset import get_loader_stats as get_loader_stats

config = {"train": {'batch_size': 1, 'shuffle': True, 'num_workers': 4},
        "valid": {'batch_size': 1, 'shuffle': False, 'num_workers': 4},
        "test": {'batch_size': 1, 'shuffle': False, 'num_workers': 4}}

base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
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
    unique_words.update(caption)

# Count the number of times that each word appears in the dataset
word_count = {}
from collections import Counter
word_count = Counter([word for caption in list_words for word in caption])
# total/(num_clases * count_word)

total = sum(word_count.values())
num_classes = len(unique_words)

NUM_WORDS = len(unique_words)
unique_words = ['<SOS>', '<EOS>', '<PAD>'] + sorted(list(unique_words))
idx2word = {k: v for k, v in enumerate(unique_words)}
word2idx = {v: k for k, v in enumerate(unique_words)}

word_weights = {word2idx[word]: total/(num_classes * count_word) for word, count_word in word_count.items()}

weights = torch.tensor([weight for idx, weight in sorted(word_weights.items(), key=lambda x: x[0])])
print(weights)



train_loader = get_loader_stats('train', config)
valid_loader = get_loader_stats('valid', config)
test_loader = get_loader_stats('test', config)

length = []
imgs = []
num_words = []
for img, caption, caption_words in tqdm(train_loader):
    imgs.append(img.size())
    num_words.append(len(caption_words))
    length.append((len(caption)))

print("Train Data")
print("\nCaptions per character")
print(f'Average length of captions: {sum(length)/len(length)}')
print(f'Maximum length of captions: {max(length)}')
print(f'Minimum length of captions: {min(length)}')
print(f'Median length of captions: {np.median(length)} +- {np.std(length)}')
print("\nCaptions per word")
print(f'Average length of captions: {sum(num_words)/len(num_words)}')
print(f'Maximum length of captions: {max(num_words)}')
print(f'Minimum length of captions: {min(num_words)}')
print(f'Median length of captions: {np.median(num_words)} +- {np.std(num_words)}')
print("\nImages")
print(f'Mean image width: {np.mean([img[2] for img in imgs])} +- {np.std([img[2] for img in imgs])}')
print(f'Mean image height: {np.mean([img[3] for img in imgs])} +- {np.std([img[3] for img in imgs])}')
print(f"Max image width: {max([img[2] for img in imgs])}")
print(f"Max image height: {max([img[3] for img in imgs])}")
print(f"Min image width: {min([img[2] for img in imgs])}")
print(f"Min image height: {min([img[3] for img in imgs])}")


length = []
imgs = []
num_words = []
for img, caption, caption_words in tqdm(valid_loader):
    imgs.append(img.size())
    num_words.append(len(caption_words))
    length.append((len(caption)))


print("Validation")
print("\nCaptions per character")
print(f'Average length of captions: {sum(length)/len(length)}')
print(f'Maximum length of captions: {max(length)}')
print(f'Minimum length of captions: {min(length)}')
print(f'Median length of captions: {np.median(length)} +- {np.std(length)}')
print("\nCaptions per word")
print(f'Average length of captions: {sum(num_words)/len(num_words)}')
print(f'Maximum length of captions: {max(num_words)}')
print(f'Minimum length of captions: {min(num_words)}')
print(f'Median length of captions: {np.median(num_words)} +- {np.std(num_words)}')
print("\nImages")
print(f'Mean image width: {np.mean([img[2] for img in imgs])} +- {np.std([img[2] for img in imgs])}')
print(f'Mean image height: {np.mean([img[3] for img in imgs])} +- {np.std([img[3] for img in imgs])}')
print(f"Max image width: {max([img[2] for img in imgs])}")
print(f"Max image height: {max([img[3] for img in imgs])}")
print(f"Min image width: {min([img[2] for img in imgs])}")
print(f"Min image height: {min([img[3] for img in imgs])}")

length = []
imgs = []
num_words = []
for img, caption, caption_words in tqdm(test_loader):
    imgs.append(img.size())
    num_words.append(len(caption_words))
    length.append((len(caption)))


print("Test")
print("\nCaptions per character")
print(f'Average length of captions: {sum(length)/len(length)}')
print(f'Maximum length of captions: {max(length)}')
print(f'Minimum length of captions: {min(length)}')
print(f'Median length of captions: {np.median(length)} +- {np.std(length)}')
print("\nCaptions per word")
print(f'Average length of captions: {sum(num_words)/len(num_words)}')
print(f'Maximum length of captions: {max(num_words)}')
print(f'Minimum length of captions: {min(num_words)}')
print(f'Median length of captions: {np.median(num_words)} +- {np.std(num_words)}')
print("\nImages")
print(f'Mean image width: {np.mean([img[2] for img in imgs])} +- {np.std([img[2] for img in imgs])}')
print(f'Mean image height: {np.mean([img[3] for img in imgs])} +- {np.std([img[3] for img in imgs])}')
print(f"Max image width: {max([img[2] for img in imgs])}")
print(f"Max image height: {max([img[3] for img in imgs])}")
print(f"Min image width: {min([img[2] for img in imgs])}")
print(f"Min image height: {min([img[3] for img in imgs])}")