import torch
from torch import nn
from transformers import ResNetModel
import random

class Bl_gru(nn.Module):
    def __init__(self, text_max_len, teacher_forcing_ratio, NUM_CHARS, char2idx, idx2char, dropout=0.1, rnn_layers = 1):
        super().__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(self.DEVICE)
        self.rnn = nn.GRU(512, 512, num_layers = rnn_layers)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(512, NUM_CHARS)
        self.embed = nn.Embedding(NUM_CHARS, 512)
        self.char2idx = char2idx
        self.TEXT_MAX_LEN = text_max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.rnn_layers = rnn_layers

    def unfreeze_params(self, value = False):
        for param in self.resnet.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth=None):
        batch_size = img.shape[0]
        feat = self.resnet(img) # batch, 512, 1, 1
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0).repeat(self.rnn_layers, 1, 1) # 1, batch, 512 # .repeat(2, 1, 1) # Per utilitzar mes de una layer de la gru
        start = torch.tensor(self.char2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        hidden = feat
        pred = inp
        for i in range(self.TEXT_MAX_LEN - 1): # rm <SOS>
            out, hidden = self.rnn(inp, hidden) # 1, batch, 512
            
            pred = torch.cat((pred, out), dim=0)
            
            if (ground_truth is not None) and (random.random() < self.teacher_forcing_ratio):
                out = ground_truth[:, i] # batch
                out = self.embed(out).unsqueeze(0) # 1, batch, 512
            else:
                out = out.permute(1, 0, 2) # batch, seq, 512
                out = self.dropout(out)
                out = self.proj(out) # batch, seq,  NUM_WORDS
                out = out.permute(1, 0, 2) # seq, batch, NUM_WORDS
                _, out = out.max(2) # seq, batch
                out = self.embed(out) # seq, batch, 512
                out = self.dropout(out)

            inp = out
        
        res = pred.permute(1, 0, 2) # batch, seq, 512
        res = self.dropout(res)
        res = self.proj(res) # batch, seq, 80 
        res = res.permute(0, 2, 1) # batch, 80, seq

        return res


