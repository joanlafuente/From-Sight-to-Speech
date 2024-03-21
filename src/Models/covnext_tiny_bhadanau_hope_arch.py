import torch
from torch import nn
from torchvision.models import convnext_tiny
import pandas as pd
import random


# Put the require grad to fals

class covnext_tiny_bhadanau_hope(nn.Module):
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
        self.bahdanau = BahdanauAttention(768, rnn_layers, device=self.DEVICE) #
        
    def unfreeze_params(self, value = False):
        for param in self.convnext_tiny.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth = None):
        batch_size = img.shape[0]
        feature_maps_convx = self.convnext_tiny.features(img) # batch, 768, 7, 7

        
        avg_pool_convnx = self.convnext_tiny.avgpool(feature_maps_convx) # batch, 768, 1, 1
        feature_maps_convx = feature_maps_convx.view(batch_size, 768, -1) # badhanau
        
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
        pred = self.proj(pred) # 1, batch, NUM_WORDS
        
        out = start_embeds
        
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
            out, (h0, c0) = self.RNN(out, (h0, c0)) # 1, batch, 512
            
            # Cross attention
            weights = self.bahdanau(h0, feature_maps_convx)
            context = torch.bmm(weights, feature_maps_convx.permute(0, 2, 1)).permute(1, 0, 2) # 1, batch, 512

            out = torch.cat((context, out), dim = 2) # seq, batch, 768 + 768 = 1536
            #pred = torch.cat((pred, out), dim=0)
            
            #out = out.permute(1, 0, 2) # batch, seq, 768
            out = self.proj(out) # seq, batch, NUM_WORDS
            pred = torch.cat((pred, out), dim = 0)
            #out = out.permute(1, 0, 2) # seq, batch, NUM_WORDS
            _, out = out.max(2) # seq, batch
            out = self.embed(out) # seq, batch, 512

        return pred
    
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
    def forward(self, hidden, encoder_output):
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
    model = covnext_tiny_bhadanau_hope(text_max_len=38, teacher_forcing_ratio=1, NUM_WORDS=NUM_WORDS, word2idx=word2idx)

    img = torch.randn(2, 3, 224, 224)
    res = model(img)
    print(res.shape) # batch, NUM_WORDS, seq
