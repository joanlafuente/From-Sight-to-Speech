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
    def __init__(self, TOTAL_MAX_WORDS, teacher_forcing_ratio, device, rnn_layers = 3, dropout=0.1):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(device)
        self.layers = rnn_layers
        self.RNN = nn.LSTM(512*2, 512, num_layers=self.layers)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(512, NUM_WORDS)
        self.embed = nn.Embedding(NUM_WORDS, 512)
        self.word2idx = word2idx
        self.DEVICE = device
        self.TOTAL_MAX_WORDS = TOTAL_MAX_WORDS
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.attention = BahdanauAttention(512, self.layers, self.DEVICE) # 512 is the hidden size of the LSTM, 3 is the number of layers of the LSTM

    def unfreeze_params(self, value = False):
        for param in self.resnet.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth = None):
        batch_size = img.shape[0]
        feat = self.resnet(img) # batch, 512, 1, 1
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512 # .repeat(2, 1, 1) # Per utilitzar mes de una layer de la gru
        
        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        
        feat = feat.repeat(self.layers, 1, 1).to(self.DEVICE) # rnn_layers, batch, 512
        
        att_weights = self.attention(feat, feat, [self.layers]*batch_size).to(self.DEVICE)
        feat_t = torch.bmm(att_weights, feat.permute(1, 0, 2)) # batch, rnn_layers, 512
        feat_t = feat_t.permute(1, 0, 2) # 3, batch, 512
        
        # aixo ho recomana el copilot i pot tenir sentit
        inp = torch.cat((feat_t, start_embeds), dim=2) # rnn_layers, batch, 1024
        h0 = feat_t
        c0 = feat_t

        pred = start_embeds # 1, batch, 512
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
            # print("inp", inp.shape)
            out, (h0, c0) = self.RNN(inp, (h0, c0)) # 1, batch, 512
            # print("out", out.shape)
            pred = torch.cat((pred, out), dim=0) # N, batch, 512
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
            
            att_weights = self.attention(out, feat, [self.layers]*batch_size).to(self.DEVICE)
            feat_t = torch.bmm(att_weights, feat.permute(1, 0, 2)) # batch, 1, 512
            feat_t = feat_t.permute(1, 0, 2) # 1, batch, 512
            feat_t = torch.cat((feat_t, out), dim=2) # 1, batch, 1024


            inp = feat_t
            # inp = torch.cat((inp, out[-1:]), dim=0) # N, batch, 512 # N es el numero de words que s'han predit (la longitud de la seq)

        res = pred.permute(1, 0, 2) # batch, seq, 512
        res = self.dropout(res)
        # print("res", res.size())
        res = self.proj(res) # batch, seq,  NUM_WORDS
        # print("res", res.size())
        res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
        # print("res", res.size())
        return res

# Standard Bahdanau Attention
from torch import nn
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, decoder_layer, device = 'cuda'):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.decoder_layer = decoder_layer
        self.softmax = nn.Softmax(dim=0)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.hidden_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.DEVICE = device

    # hidden: 1, batch, features
    # encoder_output: batch, time_step, features
    def score(self, hidden, encoder_output):
        hidden = hidden.permute(1, 2, 0) # batch, features, layers
        addMask = torch.FloatTensor([1/self.decoder_layer] * self.decoder_layer).view(1, self.decoder_layer, 1).to(self.DEVICE)
        addMask = torch.cat([addMask] * hidden.shape[0], dim=0)
        hidden = torch.bmm(hidden, addMask) # batch, feature, 1
        hidden = hidden.permute(0, 2, 1) # batch, 1, features
        hidden_attn = self.hidden_proj(hidden) # b, 1, f
        #hidden_attn = hidden_attn.permute(1, 0, 2) # batch, 1, features

        # encoder_output # b, t, features
        encoder_output_attn = self.encoder_output_proj(encoder_output) # t, b, f
        encoder_output_attn = encoder_output_attn.permute(1, 0, 2) # b, t, f
        res_attn = self.tanh(encoder_output_attn + hidden_attn) # b, t, f
        #res_attn = self.tanh(encoder_output + hidden_attn) # b, t, f
        out_attn = self.out(res_attn) # b, t, 1
        out_attn = out_attn.squeeze(2) # b, t
        return out_attn

    # hidden: b, f  encoder_output: t, b, f  enc_len: numpy
    def forward(self, hidden, encoder_output, enc_len):
        # prev_attention will not be used aqui
        encoder_output = encoder_output.transpose(0, 1) # b, t, f
        attn_energy = self.score(hidden, encoder_output) # b, t

        attn_weight = torch.zeros(attn_energy.shape)
        for i, le in enumerate(enc_len):
            attn_weight[i, :le] = self.softmax(attn_energy[i, :le])
        return attn_weight.unsqueeze(2)
    

from torchvision.models import resnet18
# Import feature extractor from torchvision
from torchvision.models.feature_extraction import create_feature_extractor

class LSTM_attention(nn.Module):
    def __init__(self, TOTAL_MAX_WORDS=5000, teacher_forcing_ratio=0, device="cuda", rnn_layers = 3, dropout=0.1):
        super().__init__()
        # Get the input of the average pooling layer
        self.resnet = create_feature_extractor(
            resnet18(pretrained=True),
            return_nodes={"layer4": "output"}
        ).to(device)
        self.layers = rnn_layers
        self.RNN = nn.GRU(512, 512, num_layers=self.layers)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(512, NUM_WORDS)
        self.embed = nn.Embedding(NUM_WORDS, 512)
        self.word2idx = word2idx
        self.DEVICE = device
        self.TOTAL_MAX_WORDS = TOTAL_MAX_WORDS
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.attention = BahdanauAttention(512, self.layers, self.DEVICE) # 512 is the hidden size of the LSTM, 3 is the number of layers of the LSTM
        self.linear_att = nn.Linear(1024, 512)
    def unfreeze_params(self, value = False):
        for param in self.resnet.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth = None):
        batch_size = img.shape[0]
        feat = self.resnet(img)["output"] # batch, 512, 7, 7
        feat = feat.view(feat.shape[0], feat.shape[1], -1)
        feat = feat.permute(0, 2, 1) # batch, 49, 512

        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)

        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds

        # Average pooling for the first hidden state and cell state
        feat_in = feat.mean(1).unsqueeze(1).repeat(1, self.layers, 1) # batch, layers, 512
        feat_in = feat_in.permute(1, 0, 2).contiguous() # 3, layers, 512
        # Els contiguous son per aixo:
        # In PyTorch, tensors can be either contiguous or non-contiguous. A tensor is contiguous if it occupies a single, unbroken block of memory.
        h = feat_in 
        # c = feat_in


        pred = inp # 1, batch, 512
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
            # print("inp", inp.shape)
            # out, (h, c) = self.RNN(inp, (h, c)) # 1, batch, 512
            out, (h) = self.RNN(inp, (h)) # 1, batch, 512
            # print("out", out.shape)
            pred = torch.cat((pred, out), dim=0) # N, batch, 512
            if (ground_truth is not None) and (random.random() < self.teacher_forcing_ratio):
                out = ground_truth[:, i] # batch
                out = self.embed(out).unsqueeze(0) # 1, batch, 512
            else:
                out = out.permute(1, 0, 2) # batch, 1, 512
                #out = self.dropout(out)
                out = self.proj(out) # batch, 1,  NUM_WORDS
                _, out = out.max(2) # batch, 1
                # print(out.size())
                out = self.embed(out) # batch, 1, 512
                out = out.permute(1, 0, 2) # 1, batch, NUM_WORDS
                #out = self.dropout(out)
            inp = out

            att_weights_h = self.attention(h, feat, [self.layers]*batch_size).permute(0, 2, 1).to(self.DEVICE) # 1, batch, 49
            # Weighted sum of encoder outputs
            h_att = torch.matmul(att_weights_h, feat)#.repeat(1, self.layers, 1) # batch, n_layers, 512
            h_att = h_att.permute(1, 0, 2).contiguous() # n_layers, batch, 512
            h = torch.cat((h, h_att), dim=2) # n_layers, batch, 1024
            h = h.permute(1, 0, 2).contiguous() # batch, n_layers, 1024
            h = self.linear_att(h)
            h = h.permute(1, 0, 2).contiguous() # n_layers, batch, 512

            # att_weights_c = self.attention(c, feat, [self.layers]*batch_size).permute(0, 2, 1).to(self.DEVICE) # 1, batch, 49
            # c = torch.matmul(att_weights_c, feat).repeat(1, self.layers, 1) # batch, n_layers, 512
            # c = c.permute(1, 0, 2).contiguous().to(self.DEVICE) # n_layers, batch, 512


    
        res = pred.permute(1, 0, 2) # batch, seq, 512
        #res = self.dropout(res)
        # print("res", res.size())
        res = self.proj(res) # batch, seq,  NUM_WORDS
        # print("res", res.size())
        res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
        # print("res", res.size())
        return res
    
# resnet = create_feature_extractor(
#             resnet18(pretrained=True),
#             return_nodes={"layer4": "output"}
# )
# feat = resnet(torch.randn(4, 3, 224, 224))["output"]
# feat = feat.view(feat.shape[0], feat.shape[1], -1)
# feat = feat.permute(0, 2, 1) # batch, 49, 512
# print(feat.size())

# # Average pooling
# print(feat.mean(1).unsqueeze(1).repeat(1, 3, 1).size())

# attention = BahdanauAttention(512, 3, "cpu") # 512 is the hidden size of the LSTM, 3 is the number of layers of the LSTM

# att_weights = attention(torch.randn(3, 4, 512), feat, [3]*4).permute(0, 2, 1) # 1, batch, 49
# h = torch.matmul(att_weights, feat).repeat(1, 3, 1) # batch, n_layers, 512

# print(h.size())
