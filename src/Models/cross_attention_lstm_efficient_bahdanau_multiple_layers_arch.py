import torch
from torch import nn
from transformers import EfficientNetModel
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

class cross_attention_lstm_efficient_bahdanau_multiple_layers(nn.Module):
    def __init__(self, text_max_len, teacher_forcing_ratio, NUM_WORDS, word2idx, dropout = 0, rnn_layers = 2, return_attn = False, idx2word = None):
        super().__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = EfficientNetModel.from_pretrained('google/efficientnet-b3').to(self.DEVICE)
        self.RNN = nn.LSTM(1536, 1536, num_layers = rnn_layers)
        self.proj = nn.Linear(1536, NUM_WORDS)
        self.embed = nn.Embedding(NUM_WORDS, 1536)
        self.word2idx = word2idx
        self.TOTAL_MAX_WORDS = text_max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.attention = BahdanauAttention(1536, rnn_layers, device=self.DEVICE)
        self.h_att = nn.ModuleList([nn.Linear(1536*2, 1536).to(self.DEVICE) for i in range(rnn_layers)])
        self.dropout = nn.Dropout(dropout)
        self.return_attn = return_attn


    def unfreeze_params(self, value = False):
        for param in self.resnet.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth = None):
        batch_size = img.shape[0]
        feat = self.resnet(img) # batch, 512, 1, 1

        img_channels = feat.last_hidden_state # batch, 512, 7, 7
        img_channels = img_channels.view(batch_size, 1536, -1) # batch, 512, 49

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
        if self.return_attn:
            all_weights = None
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
            
            weights = self.attention.score(h0, img_channels) # batch, 1, 49
            attention_image = torch.bmm(weights, img_channels.permute(0, 2, 1)).permute(1, 0, 2) # batch, 1, 512

            for i in range(self.RNN.num_layers):
                new_h = torch.tanh(self.h_att[i](torch.cat((attention_image, h0[i].unsqueeze(0)), dim=2)))
                if i == 0:
                    h0_att = new_h
                else:
                    h0_att = torch.cat((h0_att, new_h), dim=0)
            h0 = h0_att

            if self.return_attn:         
                if all_weights is None:
                    all_weights = weights.unsqueeze(0)
                else:
                    all_weights = torch.cat((all_weights, weights.unsqueeze(0)), dim=0)


        if not self.return_attn:
            res = pred.permute(1, 0, 2) # batch, seq, 512
            #res = self.dropout(res)
            res = self.proj(res) # batch, seq,  NUM_WORDS
            res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
            return res
        else:
            res = pred.permute(1, 0, 2) # batch, seq, 512
            #res = self.dropout(res)
            res = self.proj(res) # batch, seq,  NUM_WORDS
            res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
            return res, all_weights

    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, decoder_layer, device = 'cuda'):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.decoder_layer = decoder_layer
        self.softmax = nn.Softmax(dim=2)
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

        encoder_output = encoder_output.permute(2, 0, 1)
        encoder_output_attn = self.encoder_output_proj(encoder_output) # t, b, f
        encoder_output_attn = encoder_output_attn.permute(1, 0, 2) # b, t, f
        res_attn = self.tanh(encoder_output_attn + hidden_attn) # b, t, f
        #res_attn = self.tanh(encoder_output + hidden_attn) # b, t, f
        out_attn = self.out(res_attn) # b, t, 1
        out_attn = out_attn.permute(0, 2, 1) # b, 1, t
        out_attn = self.softmax(out_attn) # b, 1, t
        return out_attn



if __name__ == "__main__":
    import numpy as np
    base_path = "/fhome/gia07/Image_captioning/src/Dataset/"
    img_path = f'{base_path}Images/'
    cap_path = f'{base_path}captions.txt'
    path_partitions = f'{base_path}flickr8k_partitions.npy'
    data = pd.read_csv(cap_path)

    # Open partitions file
    NUM_CAPTIONS = 5
    partitions = np.load(path_partitions, allow_pickle=True).item()

    s = []
    for idx in partitions['train']:
        s.extend([(idx * NUM_CAPTIONS) + i for i in range(5)])
    idx = np.array(s)
    result = data.iloc[idx]

    captions = result.caption.apply(lambda x: x.lower()).values

    unique_words = set()
    list_words = []
    for i in range(len(captions)):
        caption = captions[i]
        caption = caption.split()
        list_words.append(caption)
        unique_words.update(caption)

    # Count the number of times that each word appears in the dataset
    word_count = {}
    from collections import Counter
    word_count = Counter([word for caption in list_words for word in caption])
    # total/(num_clases * count_word)

    total = sum(word_count.values())
    num_classes = len(unique_words)

    NUM_WORDS = len(unique_words)
    unique_words = ['<SOS>', '<EOS>', '<PAD>'] + sorted(list(unique_words))
    idx2word = {k: v for k, v in enumerate(unique_words)}
    word2idx = {v: k for k, v in enumerate(unique_words)}

    # Init model
    model = cross_attention_lstm_efficient_bahdanau_multiple_layers(text_max_len=38, teacher_forcing_ratio=0, NUM_WORDS=NUM_WORDS, word2idx=word2idx)

    img = torch.randn(2, 3, 224, 224)
    res = model(img)
    print(res.shape) # batch, NUM_WORDS, seq
