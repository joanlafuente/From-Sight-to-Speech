from tqdm import tqdm
import numpy as np

from src.data_utils.dataset import Dataset, get_loader_stats
config = {"train": {'batch_size': 1, 'shuffle': True, 'num_workers': 4},
        "valid": {'batch_size': 1, 'shuffle': False, 'num_workers': 4},
        "test": {'batch_size': 1, 'shuffle': False, 'num_workers': 4}}

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