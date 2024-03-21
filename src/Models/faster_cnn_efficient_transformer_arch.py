import torch
from torch import nn
from transformers import EfficientNetModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pandas as pd
import random

class faster_cnn_efficient_transformer(nn.Module):
    def __init__(self, text_max_len, teacher_forcing_ratio, NUM_WORDS, word2idx, idx2word, dropout = 0, rnn_layers = 1, return_attn = False):
        super().__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.efficient = EfficientNetModel.from_pretrained('google/efficientnet-b0').to(self.DEVICE)
        self.fasterrcnn = fasterrcnn_resnet50_fpn(pretrained=True).to(self.DEVICE)
        
        dim_embedding = 1280
        transformer_heads = 1
        transformer_layers = 1
        
        transformer_layer = nn.TransformerDecoderLayer(d_model=dim_embedding, nhead=transformer_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(transformer_layer, num_layers=transformer_layers)
        
        self.embed = nn.Embedding(NUM_WORDS, 1280)
        self.proj = nn.Linear(1280, NUM_WORDS)
        self.text_max_len = text_max_len
        self.word2idx = word2idx

    def unfreeze_params(self, value = False):
        for param in self.efficient.parameters():
            param.requires_grad = value

    def forward(self, img, regions, ground_truth = None):
        batch_size = img.shape[0]
        feat = self.efficient(img)

        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 1280 # .repeat(2, 1, 1) # Per utilitzar mes de una layer de la gru
        
        # for each region create a black image and put the region in its place then pass it through the efficient
        # and get the features of each region
        
        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        prev_word = self.embed(start) # 512
        prev_words = prev_word.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = prev_words
        pred = prev_words
        
        regions = regions.permute(1, 0, 2).to(self.DEVICE)
        
        # print(prev_words.shape, feat_regions.shape) # 1, 5, 1280
        for i in range(self.text_max_len - 1): # rm <SOS>
            # print(inp.shape, regions.shape) # -> [1, 5, 1280]) torch.Size([5, 6, 1280]
            output_transformer = self.transformer_decoder(inp, regions) # 1, batch, 1280
            
            pred = torch.cat((pred, output_transformer[-1:, :, :]), dim=0)
        
            output_transformer = output_transformer[-1:, :, :].permute(1, 0, 2) # batch, num_decoder_layers, out_cnn
            output_transformer = self.proj(output_transformer) # batch, num_decoder_layers, NUM_WORDS
            output_transformer = output_transformer.permute(1, 0, 2) # num_decoder_layers, batch, NUM_WORDS
            _, output_transformer = output_transformer.max(2) # num_decoder_layers, batch
            prev_word = self.embed(output_transformer)
            
            inp = torch.cat((inp, prev_word), dim=0)
            
        # output_transformer has shape [num_words_sentence, batch, size_vocab]
        
        # print(pred.shape) # 6, 38, 1280 -> batch, num_regions, d_model
        pred = pred.permute(1, 0, 2) # num_regions, batch, d_model
        pred = self.proj(pred)
        pred = pred.permute(0, 2, 1) # batch, num_regions, d_model
        return pred
        
        

if __name__ == "__main__":
    import numpy as np
    base_path = "/fhome/gia07/Image_captioning/src/Dataset/"
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

    # Init model
    model = faster_cnn_efficient_transformer(text_max_len = 38, teacher_forcing_ratio = 0, NUM_WORDS = NUM_WORDS, word2idx = word2idx, idx2word = idx2word, dropout = 0)
            # VIT_transformer_updated(text_max_len=38, teacher_forcing_ratio=0, NUM_WORDS=NUM_WORDS, word2idx=word2idx)

    img = torch.randn(2, 3, 224, 224)
    caption = torch.randint(0, NUM_WORDS, (2, 38))
    res = model(img, regions)
    print(res.shape) # batch, NUM_WORDS, seq
