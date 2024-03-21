import torch
from torch import nn
from transformers import ResNetModel
import random

class Bl_gru(nn.Module):
    def __init__(self, text_max_len, teacher_forcing_ratio=0, dropout=0):
        super().__init__()
        chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        NUM_CHAR = len(chars)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(self.DEVICE)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)
        self.char2idx = {v: k for k, v in enumerate(chars)}
        self.TEXT_MAX_LEN = text_max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout = dropout

    def unfreeze_params(self, value = False):
        for param in self.resnet.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth=None):
        batch_size = img.shape[0]
        feat = self.resnet(img) # batch, 512, 1, 1
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512 # .repeat(2, 1, 1) # Per utilitzar mes de una layer de la gru
        start = torch.tensor(self.char2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        hidden = feat
        pred = inp
        for i in range(self.TEXT_MAX_LEN - 1): # rm <SOS>
            out, hidden = self.gru(inp, hidden) # 1, batch, 512
            
            pred = torch.cat((pred, out), dim=0)
            
            if (ground_truth is not None) and (random.random() < self.teacher_forcing_ratio):
                out = ground_truth[:, i] # batch
                out = self.embed(out).unsqueeze(0) # 1, batch, 512
            else:
                out = out.permute(1, 0, 2) # batch, seq, 512
                out = self.proj(out) # batch, seq,  NUM_WORDS
                out = out.permute(1, 0, 2) # seq, batch, NUM_WORDS
                _, out = out.max(2) # seq, batch
                out = self.embed(out) # seq, batch, 512
            
            inp = out
        
        res = inp.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res) # batch, seq, 80 
        res = res.permute(0, 2, 1) # batch, 80, seq
        return res
