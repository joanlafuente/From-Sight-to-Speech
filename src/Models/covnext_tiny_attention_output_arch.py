import torch
from torch import nn
from torchvision.models import convnext_tiny
import pandas as pd
import random


# Put the require grad to fals

class covnext_tiny_attention_output(nn.Module):
    def __init__(self, text_max_len, teacher_forcing_ratio, NUM_WORDS, word2idx, dropout = 0, rnn_layers = 2, return_attn = False, idx2word = None):
        super().__init__()
        self.convnext_tiny = convnext_tiny(weights = 'IMAGENET1K_V1')
        self.num_layers_RNN = rnn_layers
        self.RNN = nn.LSTM(768, 768, num_layers = self.num_layers_RNN)
        self.proj = nn.Linear(768 * 2, NUM_WORDS) # projection to the vocabulary
        self.embed = nn.Embedding(NUM_WORDS, 768)
        self.word2idx = word2idx
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TOTAL_MAX_WORDS = text_max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.cross_attention = CrossAttention(768) #
        
    def unfreeze_params(self, value = False):
        for param in self.convnext_tiny.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth = None):
        batch_size = img.shape[0]
        feature_maps_convx = self.convnext_tiny.features(img) # batch, 768, 7, 7
        avg_pool_convnx = self.convnext_tiny.avgpool(feature_maps_convx) # batch, 768, 1, 1

        avg_pool_convnx = avg_pool_convnx.squeeze(2) # batch, 768, 1
        avg_pool_convnx_1 = avg_pool_convnx.permute(2, 0, 1) # 1, batch, 768
        avg_pool_convnx = avg_pool_convnx_1.repeat(self.num_layers_RNN, 1, 1)
        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        
        start_embed = self.embed(start) # 768
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        
        # Init the hidden state
        h0 = avg_pool_convnx
        c0 = torch.zeros_like(h0)
        
        pred = torch.cat((avg_pool_convnx_1, start_embeds), dim = 2) # 1, batch, 768 + 768 = 1536
        
        out = start_embeds
        
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
            out, (h0, c0) = self.RNN(out, (h0, c0)) # 1, batch, 512
            
            # Cross attention
            context = self.cross_attention(feature_maps_convx, h0[-1]) # batch, 768

            context = context.unsqueeze(0) # 1, batch, 768
            out = torch.cat((context, out), dim = 2) # seq, batch, 768 + 768 = 1536
            pred = torch.cat((pred, out), dim=0)
            
            out = out.permute(1, 0, 2) # batch, seq, 768
            out = self.proj(out) # batch, seq,  NUM_WORDS
            out = out.permute(1, 0, 2) # seq, batch, NUM_WORDS
            _, out = out.max(2) # seq, batch
            out = self.embed(out) # seq, batch, 512
            
            
        res = pred.permute(1, 0, 2) # batch, seq, 512
        # print("res", res.size())
        res = self.proj(res) # batch, seq,  NUM_WORDS
        # print("res", res.size())
        res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
        # print("res", res.size())
        return res
    
class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Make the output of the encoder consistent with the decoder
        self.q = nn.Linear(hidden_size, hidden_size) # 768 a 768
        self.k = nn.Linear(hidden_size, hidden_size) # 768 a 768
        self.v = nn.Linear(hidden_size, hidden_size) # 768 a 768

    def scoringDot(self, keys, query):
        # keys: batch, 49, 768
        # query: batch, 768
        query = query.unsqueeze(2) # batch, 768, 1
        result = torch.bmm(keys, query) # batch, 49, 1
        result = result.squeeze(2) # batch, 49
        weights = result.softmax(1) # batch, 49
        return weights.unsqueeze(1) # batch, 1, 49  This represents the weights of each channel
        
    def forward(self, channel_img, last_hidden_lstm):
        # channel_img: batch, 768, 7, 7
        # output_lstm: batch, 768
        channel_img = channel_img.view(channel_img.shape[0], channel_img.shape[1], -1)
        channel_img = channel_img.permute(0, 2, 1) # batch, 49, 768 # 49 is the number of channels
        
        query   = torch.tanh(self.q(last_hidden_lstm)) # batch, 768
        keys    = torch.tanh(self.k(channel_img)) # batch, 49, 768
        values  = torch.tanh(self.v(channel_img)) # batch, 49, 768
        
        weights = self.scoringDot(keys, query) # batch, 1, 49
        
        # Multiply each channel by its weight
        context = torch.bmm(weights, values) # batch, 1, 768
        context = context.squeeze(1) # batch, 768
        return context
        
        