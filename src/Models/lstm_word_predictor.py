import torch
from torch import nn
from transformers import ResNetModel
import pandas as pd
import random

base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
cap_path = f'{base_path}captions.txt'
data = pd.read_csv(cap_path)

# Unique words in the dataset
unique_words = set()
captions = data.caption.apply(lambda x: x.lower()).values
for i in range(len(data)):
    caption = captions[i]
    caption = caption.split()
    unique_words.update(caption)

unique_words = ['<SOS>', '<EOS>', '<PAD>'] + list(unique_words)
NUM_WORDS = len(unique_words)
idx2word = {k: v for k, v in enumerate(unique_words)}
word2idx = {v: k for k, v in enumerate(unique_words)}
TOTAL_MAX_WORDS = 38

class Teacher_img_to_word_LSTM(nn.Module):
    def __init__(self, TOTAL_MAX_WORDS, teacher_forcing_ratio, DEVICE):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        self.RNN = nn.LSTM(512, 512, num_layers=3)
        self.proj = nn.Linear(512, NUM_WORDS)
        self.embed = nn.Embedding(NUM_WORDS, 512)
        self.word2idx = word2idx
        self.DEVICE = DEVICE
        self.TOTAL_MAX_WORDS = TOTAL_MAX_WORDS
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, img, ground_truth = None):
        batch_size = img.shape[0]
        feat = self.resnet(img) # batch, 512, 1, 1
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512 # .repeat(2, 1, 1) # Per utilitzar mes de una layer de la gru
        feat = feat.repeat(3, 1, 1)
        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        # print(start.shape, self.word2idx['<SOS>'])
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        h0 = feat
        c0 = feat

        pred = inp
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
            out, (_, _) = self.RNN(inp, (h0, c0)) # 1, batch, 512
            # print("out", out.shape)
            pred = torch.cat((pred, out[-1:]), dim=0)
            if (ground_truth is not None) and (random.random() < self.teacher_forcing_ratio):
                out = ground_truth[:, i] # batch
                out = self.embed(out).unsqueeze(0) # 1, batch, 512
            else:
                out = out.permute(1, 0, 2) # batch, seq, 512
                out = self.proj(out) # batch, seq,  NUM_WORDS
                out = out.permute(1, 0, 2) # seq, batch, NUM_WORDS
                _, out = out.max(2) # seq, batch
                out = self.embed(out) # seq, batch, 512
            
            # inp = out
            inp = torch.cat((inp, out[-1:]), dim=0) # N, batch, 512 # N es el numero de words que s'han predit (la longitud de la seq)

        res = pred.permute(1, 0, 2) # batch, seq, 512
        # print("res", res.size())
        res = self.proj(res) # batch, seq,  NUM_WORDS
        # print("res", res.size())
        res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
        # print("res", res.size())
        return res