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

        self.fc_h = nn.Linear(512, 512)
        self.fc_c = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, 512)
        self.V_h = nn.Linear(1024, 512)
        self.V_c = nn.Linear(1024, 512)
        self.tanh = nn.Tanh()

        self.word2idx = word2idx
        self.DEVICE = DEVICE
        self.TOTAL_MAX_WORDS = TOTAL_MAX_WORDS
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def unfreeze_params(self, value = False):
        for param in self.resnet.parameters():
            param.requires_grad = value

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
        h = feat
        c = feat
        pred = inp
        
        # print("inp", inp.size(), "h", h.size(), "c", c.size())
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
            # print("h", h.size(), "c", c.size())
            out, (h, c) = self.RNN(inp, (h, c)) # 1, batch, 512
            # print("h", h.size(), "c", c.size())
            # print("out", out.shape)
            pred = torch.cat((pred, out[-1:]), dim=0)

            out = out.permute(1, 0, 2) # batch, seq, 512
            out = self.proj(out) # batch, seq,  NUM_WORDS
            out = out.permute(1, 0, 2) # seq, batch, NUM_WORDS
            _, out = out.max(2) # seq, batch
            out = self.embed(out) # seq, batch, 512
            inp = out

            # Attention

            # print("h", h.size(), "c", c.size())
            # print("out", out.size())
            h = h.permute(1, 0, 2) # batch, 3, 512
            c = c.permute(1, 0, 2) # batch, 3, 512

            h_att = self.fc_h(h) # batch, 3, 512
            c_att = self.fc_c(c) # batch, 3, 512

            out = out[-1:] # 1, batch, 512
            out = out.permute(1, 0, 2) # batch, 1, 512
            out_att = self.fc_out(out) 

            out_att = out_att.repeat(1, 3, 1) # batch, 3, 512
            # print("h_att", h_att.size(), "out_att", out_att.size())

            # print("concat", torch.cat((h_att, out_att), dim=2).size())
            att_h = self.V_h(self.tanh(torch.cat((h_att, out_att), dim=2)))
            att_c = self.V_c(self.tanh(torch.cat((c_att, out_att), dim=2)))

            # print("att_h", att_h.size(), "att_c", att_c.size())

            att_h = torch.softmax(att_h, dim=2)
            att_c = torch.softmax(att_c, dim=2)

            att_h = att_h.permute(1, 0, 2)
            att_c = att_c.permute(1, 0, 2)

            h = feat * att_h
            c = feat * att_c

            # print("h", h.size(), "c", c.size())
            
            
        res = pred.permute(1, 0, 2) # batch, seq, 512
        # print("res", res.size())
        res = self.proj(res) # batch, seq,  NUM_WORDS
        # print("res", res.size())
        res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
        # print("res", res.size())
        return res