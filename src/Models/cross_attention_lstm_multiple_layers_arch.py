import torch
from torch import nn
from transformers import ResNetModel
import pandas as pd
import random

# base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
# cap_path = f'{base_path}captions.txt'
# data = pd.read_csv(cap_path)

# # Unique words in the dataset
# unique_words = set()
# captions = data.caption.apply(lambda x: x.lower()).values
# for i in range(len(data)):
#     caption = captions[i]
#     caption = caption.split()
#     unique_words.update(caption)

# unique_words = ['<SOS>', '<EOS>', '<PAD>'] + sorted(list(unique_words))
# NUM_WORDS = len(unique_words)
# idx2word = {k: v for k, v in enumerate(unique_words)}
# word2idx = {v: k for k, v in enumerate(unique_words)}
# TOTAL_MAX_WORDS = 38

class cross_attention_lstm_multiple_layers(nn.Module):
    def __init__(self, text_max_len, teacher_forcing_ratio, NUM_WORDS, word2idx, dropout = 0, rnn_layers = 1, return_attn = False):
        super().__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(self.DEVICE)
        self.RNN = nn.LSTM(512, 512, num_layers = rnn_layers)
        self.proj = nn.Linear(512, NUM_WORDS)
        self.embed = nn.Embedding(NUM_WORDS, 512)
        self.word2idx = word2idx
        self.TOTAL_MAX_WORDS = text_max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.cross_attention = CrossAttention(49, 512, ret_attn=return_attn) 
        self.hiddenattention2hidden = nn.Linear(512, 512)
        self.h_att = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(dropout)
        self.return_attn = return_attn


    def unfreeze_params(self, value = False):
        for param in self.resnet.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth = None):
        batch_size = img.shape[0]
        feat = self.resnet(img) # batch, 512, 1, 1

        img_channels = feat.last_hidden_state # batch, 512, 7, 7

        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512
        feat = feat.repeat(self.RNN.num_layers, 1, 1) # num_layers, batch, 512
        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        h0 = feat
        c0 = feat

        pred = inp
        out = inp
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>

            out, (h0, c0) = self.RNN(out, (h0, c0)) # 1, batch, 512
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
            
            for i in range(self.RNN.num_layers):
                h_attention = self.cross_attention(img_channels, h0[i]).unsqueeze(0) # batch, 512
                # print(h_attention.shape)
                # print(h0[i].shape)
                new_h = torch.tanh(self.h_att(torch.cat((h_attention, h0[i].unsqueeze(0)), dim=2)))
                if i == 0:
                    h0_att = new_h
                else:
                    h0_att = torch.cat((h0_att, new_h), dim=0)

            h0 = h0_att


        res = pred.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res) # batch, seq,  NUM_WORDS
        res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
        return res
    
class CrossAttention(nn.Module):
    def __init__(self, input_size, hidden_size, ret_attn=False):
        super().__init__()
        self.fc_q = nn.Linear(hidden_size, hidden_size) # 512 a 512
        self.fc_k = nn.Linear(input_size, input_size) # 49 a 49
        self.fc_v = nn.Linear(input_size, input_size) # 49 a 49
        self.ret_attn = ret_attn

    def scoringDot(self, keys, query):
        # query: batch, 512
        # keys: batch, 512, 512
        query = torch.tanh(self.fc_q(query)) # batch, 49
        keys = torch.tanh(self.fc_k(keys))  # batch, 512, 49

        query = query.unsqueeze(2) # batch, 49, 1
        keys = keys.permute(0, 2, 1)
        result = torch.bmm(keys, query) # batch, 49, 1
        result = result.squeeze(2)
        weights = (result/7).softmax(1) # batch, 49
        return weights.unsqueeze(1) # batch, 1, 49  This represents the weights of each regiion

    def forward(self, channel_img, last_hidden_lstm):
        # channel_img: batch, 512, 7, 7
        # output_lstm: batch, 512
        channel_img = channel_img.view(channel_img.shape[0], channel_img.shape[1], -1) # Batch, 512, 49

        values = torch.tanh(self.fc_v(channel_img)) # Batch, 512, 512
        weights = self.scoringDot(channel_img, last_hidden_lstm) # batch, 1, 49
        values = values.permute(0, 2, 1)
        result = torch.bmm(weights, values) # batch, 1, 512
        result = result.squeeze(1) # batch, 512
        if not self.ret_attn:
            return result
        else:
            return result, weights


