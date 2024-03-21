import torch
from torch import nn
from transformers import ResNetModel
import pandas as pd
import random


class Teacher_img_to_word_LSTM(nn.Module):
    def __init__(self, text_max_len, NUM_WORDS, word2idx, idx2word, teacher_forcing_ratio, dropout=0.1, rnn_layers = 1):
        super().__init__()

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.res50 = ResNetModel.from_pretrained('microsoft/resnet-50').to(self.DEVICE)
        self.rnn_layers = rnn_layers
        self.RNN = nn.LSTM(2048, 2048, self.rnn_layers)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(2048, NUM_WORDS)
        self.embed = nn.Embedding(NUM_WORDS, 2048)
        self.word2idx = word2idx
        self.TOTAL_MAX_WORDS = text_max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio


    def unfreeze_params(self, value = False):
        for param in self.res50.parameters():
            param.requires_grad = value
            
    def forward(self, img, ground_truth = None):
        batch_size = img.shape[0]
        feat = self.res50(img) # batch, 2048, 1, 1
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 2048 # .repeat(2, 1, 1) # Per utilitzar mes de una layer de la gru
        feat = feat.repeat(self.rnn_layers, 1, 1)
        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        # print(start.shape, self.word2idx['<SOS>'])
        start_embed = self.embed(start) # 2048
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 2048
        inp = start_embeds
        h0 = feat
        c0 = feat
        

        pred = inp
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
            out, (h0, c0) = self.RNN(inp, (h0, c0)) # 1, batch, 2048
            # print("out", out.shape)
            pred = torch.cat((pred, out[-1:]), dim=0)
            if (ground_truth is not None) and (random.random() < self.teacher_forcing_ratio):
                out = ground_truth[:, i] # batch
                out = self.embed(out).unsqueeze(0) # 1, batch, 2048
            else:
                out = out.permute(1, 0, 2) # batch, seq, 2048
                out = self.dropout(out)
                out = self.proj(out) # batch, seq,  NUM_WORDS
                out = out.permute(1, 0, 2) # seq, batch, NUM_WORDS
                _, out = out.max(2) # seq, batch
                out = self.embed(out) # seq, batch, 2048
                out = self.dropout(out)
            
            inp = out
            # inp = torch.cat((inp, out[-1:]), dim=0) # N, batch, 2048 # N es el numero de words que s'han predit (la longitud de la seq)

        res = pred.permute(1, 0, 2) # batch, seq, 2048
        res = self.dropout(res)
        # print("res", res.size())
        res = self.proj(res) # batch, seq,  NUM_WORDS
        # print("res", res.size())
        res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
        # print("res", res.size())
        return res