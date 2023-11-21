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

class image_to_word(nn.Module):
    def __init__(self, TOTAL_MAX_WORDS, DEVICE):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_WORDS)
        self.embed = nn.Embedding(NUM_WORDS, 512)
        self.word2idx = word2idx
        self.DEVICE = DEVICE
        self.TOTAL_MAX_WORDS = TOTAL_MAX_WORDS

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img) # batch, 512, 1, 1
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512 # .repeat(2, 1, 1) # Per utilitzar mes de una layer de la gru
        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        hidden = feat

        for _ in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
            out, hidden = self.gru(inp, hidden) # 1, batch, 512
            inp = torch.cat((inp, out[-1:]), dim=0) # N, batch, 512 # N es el numero de characters

        res = inp.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res) # batch, seq,  NUM_WORDS
        res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
        return res

class image_to_word_tf(nn.Module):
    def __init__(self, TOTAL_MAX_WORDS, DEVICE, NUM_WORDS, word2idx):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_WORDS)
        self.embed = nn.Embedding(NUM_WORDS, 512)
        self.word2idx = {'<SOS>':word2idx['<SOS>']}
        self.num_words = NUM_WORDS
        self.DEVICE = DEVICE
        self.TOTAL_MAX_WORDS = TOTAL_MAX_WORDS

    def forward(self, img, captions):
        batch_size = img.shape[0]
        feat = self.resnet(img)  # batch, 512, 1, 1
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)  # Reshape to (batch_size, 512)

        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start)  # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)  # 1, batch, 512
        inp = start_embeds
        hidden = feat#.repeat(1, batch_size, 1)  # Repeat the feature for each sequence in the batch

        # # Aixo es el que estava abans
        # res = []
        # for i in range(self.TOTAL_MAX_WORDS - 1):  # rm <SOS>
        #     out, hidden = self.gru(inp, hidden)
        #     out = out.squeeze(0)  # batch, 512
        #     out = self.proj(out)  # batch, NUM_WORDS
        #     res.append(out.unsqueeze(1))  # batch, 1, NUM_WORDS
        #     inp = self.embed(captions[:, i + 1]).unsqueeze(0) if random.random() < 0.5 else out.unsqueeze(0)

        # res = torch.cat(res, dim=1)  # batch, seq, NUM_WORDS
        # return res, hidden.squeeze(0)  # Remove the extra dimension

        outs = torch.zeros((batch_size, self.num_words, 1)).to(self.DEVICE)
        # Set the first output to be the start token.
        outs[:, self.word2idx['<SOS>'], 0] = 1
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
            out, hidden = self.gru(inp, hidden)
            out = self.proj(out)
            outs = torch.cat((outs, out.permute(1, 2, 0)), dim=2)
            prob, pred = torch.max(out, dim=2)
            new_input_embed = self.embed(captions[:, i + 1]).unsqueeze(0) if random.random() < 0.5 else self.embed(pred)
            inp = torch.cat((inp, new_input_embed), dim=0) # N, batch, 512
        return outs, hidden.squeeze(0) # batch, self.TOTAL_MAX_WORDS, seq