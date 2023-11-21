import torch
from torch import nn
from transformers import ResNetModel
import random

class Baseline(nn.Module):
    def __init__(self, TEXT_MAX_LEN, DEVICE):
        super().__init__()
        chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        NUM_CHAR = len(chars)
        char2idx = {v: k for k, v in enumerate(chars)}
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)
        self.char2idx = char2idx
        self.DEVICE = DEVICE
        self.TEXT_MAX_LEN = TEXT_MAX_LEN
        self.teacher_forcing_ratio = 1


    def forward(self, img, true_sentence):
        batch_size = img.shape[0]
        feat = self.resnet(img) # batch, 512, 1, 1
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512 # .repeat(2, 1, 1) # Per utilitzar mes de una layer de la gru
        # start_tk = start_tk if start_tk in self.char2idx and start_tk is not None else '<SOS>'
        start = torch.tensor(self.char2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        hidden = feat

        for i in range(self.TEXT_MAX_LEN-1): # rm <SOS>
            out, hidden = self.gru(inp, hidden) # 1, batch, 512
            inp = torch.cat((inp, out[-1:]), dim=0) # N, batch, 512 # N es el numero de characters

            if random.randrange(0, 1) < self.teacher_forcing_ratio:
                inp = self.embed(true_sentence[:, i]).unsqueeze(0) # 1, batch, 512


        res = inp.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res) # batch, seq, 80 
        res = res.permute(0, 2, 1) # batch, 80, seq
        return res


# outs = torch.zeros((batch_size, 80, 1)).to(self.DEVICE)
# # Set the first output to be the start token.
# outs[:, self.char2idx['<SOS>'], :] = 1
# for i in range(self.TEXT_MAX_LEN-1): # rm <SOS>
#     out, hidden = self.gru(inp, hidden)
#     out = self.proj(out)
#     outs = torch.cat((outs, out.permute(1, 2, 0)), dim=2)
#     prob, pred = torch.max(out, dim=2)
#     inp = torch.cat((inp, self.embed(pred)), dim=0) # N, batch, 512
# return outs # batch, 80, seq



