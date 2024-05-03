import gensim
from tqdm import tqdm
from torch import nn
import torch
import json
# Set verbosity
model = gensim.models.KeyedVectors.load_word2vec_format('/fhome/gia07/Image_captioning/src/Dataset/glove.6B.300d.txt', binary=False, no_header=True, )
print("Model loaded")


with open('/fhome/gia07/Image_captioning/src/runs-baseline/attention_convnext_tiny_bahdanau_input_multiple_layers_400_1_cap/word2idx.json') as f:
    word2idx = json.load(f)
    
idx2word = {v: k for k, v in word2idx.items()}
NUM_WORDS = len(word2idx)
embeddings = torch.randn(len(word2idx), 300)
counter = 0
for i in range(NUM_WORDS):
    word = idx2word[i]
    if word not in model:
        print(word)
        counter += 1
    else:
        embeddings[word2idx[word], :] = torch.Tensor(model[word])
print(embeddings.size())     
print(counter)
embeddings = nn.Embedding.from_pretrained(embeddings)
print(embeddings)   
    