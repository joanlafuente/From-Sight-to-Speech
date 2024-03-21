import torch
from torch import nn
from transformers import EfficientNetModel
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

unique_words = ['<SOS>', '<EOS>', '<PAD>'] + sorted(list(unique_words))
NUM_WORDS = len(unique_words)
idx2word = {k: v for k, v in enumerate(unique_words)}
word2idx = {v: k for k, v in enumerate(unique_words)}
TOTAL_MAX_WORDS = 38

class cross_attention_lstm_correct_efficient(nn.Module):
    def __init__(self, text_max_len, teacher_forcing_ratio, NUM_WORDS, word2idx, dropout = 0, rnn_layers = 1, return_attn = False):
        super().__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.efficient = EfficientNetModel.from_pretrained('google/efficientnet-b0').to(self.DEVICE)
        self.RNN = nn.LSTM(1280, 1280, num_layers = rnn_layers)
        self.proj = nn.Linear(1280, NUM_WORDS)
        self.embed = nn.Embedding(NUM_WORDS, 1280)
        self.word2idx = word2idx
        self.TOTAL_MAX_WORDS = text_max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.cross_attention = CrossAttention(49, 1280, ret_attn=return_attn) # 
        self.hiddenattention2hidden = nn.Linear(1280, 1280)
        self.h_att = nn.Linear(1280*2, 1280)
        self.c_att = nn.Linear(1280*2, 1280)
        self.dropout = nn.Dropout(dropout)
        self.return_attn = return_attn


    def unfreeze_params(self, value = False):
        for param in self.efficient.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth = None):
        batch_size = img.shape[0]
        feat = self.efficient(img) # batch, 1280, 1, 1

        img_channels = feat.last_hidden_state # batch, 1280, 7, 7

        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 1280 # .repeat(2, 1, 1) # Per utilitzar mes de una layer de la gru
        feat = feat.repeat(self.RNN.num_layers, 1, 1) # num_layers, batch, 1280
        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start) # 1280
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 1280
        inp = start_embeds
        h0 = feat
        c0 = feat

        pred = inp
        out = inp
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>

            out, (h0, c0) = self.RNN(out, (h0, c0)) # 1, batch, 1280
            pred = torch.cat((pred, out), dim=0)
            if (ground_truth is not None) and (random.random() < self.teacher_forcing_ratio):
                out = ground_truth[:, i] # batch
                out = self.embed(out).unsqueeze(0) # 1, batch, 1280
            else:
                out = out.permute(1, 0, 2) # batch, seq, 1280
                out = self.proj(out) # batch, seq,  NUM_WORDS
                out = out.permute(1, 0, 2) # seq, batch, NUM_WORDS
                _, out = out.max(2) # seq, batch
                out = self.embed(out) # seq, batch, 1280
            
            if self.return_attn:
                h0_att, weights_h0 = self.cross_attention(img_channels, h0[-1]) # batch, 1280 # h0[-1] is the last hidden state of the last layer
                h0_att = h0_att.unsqueeze(0)
                c0_att, weights_c0 = self.cross_attention(img_channels, c0[-1]) # batch, 1280 # c0[-1] is the last hidden state of the last layer
                c0_att = c0_att.unsqueeze(0)
                if i == 0:
                    concat_weights_h = weights_h0
                    concat_weights_c = weights_c0
                else:
                    concat_weights_c = torch.cat((concat_weights_c, weights_c0), dim=1)
                    concat_weights_h = torch.cat((concat_weights_h, weights_c0), dim=1)
            else:
                h0_att = self.cross_attention(img_channels, h0[-1]).unsqueeze(0) # batch, 1280 # h0[-1] is the last hidden state of the last layer
                c0_att = self.cross_attention(img_channels, c0[-1]).unsqueeze(0) # batch, 1280 # c0[-1] is the last hidden state of the last layer

            h0 = torch.tanh(self.h_att(torch.cat((h0_att, h0), dim=2)))
            c0 = torch.tanh(self.c_att(torch.cat((c0_att, c0), dim=2)))



        res = pred.permute(1, 0, 2) # batch, seq, 1280
        res = self.proj(res) # batch, seq,  NUM_WORDS
        res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
        if not self.return_attn:
            return res
        else:
            return res, concat_weights_h, concat_weights_c
    
class CrossAttention(nn.Module):
    def __init__(self, input_size, hidden_size, ret_attn=False):
        super().__init__()
        self.fc_q = nn.Linear(hidden_size, hidden_size) # 1280 a 1280
        self.fc_k = nn.Linear(input_size, input_size) # 49 a 49
        self.fc_v = nn.Linear(input_size, input_size) # 49 a 49
        self.ret_attn = ret_attn

    def scoringDot(self, keys, query):
        # query: batch, 1280
        # keys: batch, 1280, 1280
        query = torch.tanh(self.fc_q(query)) # batch, 49
        keys = torch.tanh(self.fc_k(keys))  # batch, 1280, 49

        query = query.unsqueeze(2) # batch, 49, 1
        keys = keys.permute(0, 2, 1)
        result = torch.bmm(keys, query) # batch, 49, 1
        result = result.squeeze(2)
        weights = (result/7).softmax(1) # batch, 49
        return weights.unsqueeze(1) # batch, 1, 49  This represents the weights of each regiion

    def forward(self, channel_img, last_hidden_lstm):
        # channel_img: batch, 1280, 7, 7
        # output_lstm: batch, 1280
        channel_img = channel_img.view(channel_img.shape[0], channel_img.shape[1], -1) # Batch, 1280, 49

        values = torch.tanh(self.fc_v(channel_img)) # Batch, 1280, 1280
        weights = self.scoringDot(channel_img, last_hidden_lstm) # batch, 1, 49
        values = values.permute(0, 2, 1)
        result = torch.bmm(weights, values) # batch, 1, 1280
        result = result.squeeze(1) # batch, 1280
        if not self.ret_attn:
            return result
        else:
            return result, weights


