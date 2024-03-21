import torch
from torch import nn
from torchvision.models import convnext_tiny, mobilenet_v3_large
from transformers import ResNetModel
import pandas as pd
import random


# Put the require grad to fals
# MobileNet_V3_Large_FPN_Weights
class mobilnet_roifeatures_crossattention(nn.Module):
    def __init__(self, text_max_len, teacher_forcing_ratio, NUM_WORDS, word2idx, dropout = 0, rnn_layers = 2, return_attn = False, idx2word = None):
        super().__init__()
        self.num_layers_RNN = rnn_layers
        self.RNN = nn.LSTM(1024, 1024, num_layers = self.num_layers_RNN)
        self.proj = nn.Linear(1024 * 2, NUM_WORDS) # projection to the vocabulary
        self.embed = nn.Embedding(NUM_WORDS, 1024)
        self.word2idx = word2idx
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TOTAL_MAX_WORDS = text_max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.cross_attention = CrossAttention(1024)
        
        self.cnn = convnext_tiny(weights = 'IMAGENET1K_V1')
        # self.cnn = mobilenet_v3_large(weights = 'IMAGENET1K_V2')
        # self.cnn = ResNetModel.from_pretrained('microsoft/resnet-18').to(self.DEVICE)
        self.projcnn = nn.Linear(768, 1024) # 960 Mobilenet, 512 Resnet, 768 Convnext
        

    def forward(self, info, ground_truth = None): # roi_features: batch, 1000, 1024
        roi_features, img  = info
        
        # img_features = self.cnn(img) # batch, 768, 7, 
        # img_features = img_features.pooler_output.squeeze(-1).squeeze(-1) <- Resnet
        
        img_channels = self.cnn.features(img) # <- Convnext
        img_features = self.cnn.avgpool(img_channels).squeeze(-1).squeeze(-1) # <- Convnext
        
        # print(img_features.shape)
        img_features = self.projcnn(img_features)
        
        # print(img_features.shape, roi_features.shape)
        roi_features = torch.cat((img_features.unsqueeze(1), roi_features), dim=1) # batch, 1000, 1024
        roi_features = roi_features[:, :100, :]
        
        batch_size = roi_features.shape[0]
        
        sum_reg = roi_features.sum(dim=2)        
        mask = torch.zeros_like(sum_reg)
        mask[sum_reg == 0] = float('-inf')
        
        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        
        start_embed = self.embed(start).repeat(batch_size, 1) # batch, 1024
        
        attention_SOS = self.cross_attention(roi_features, start_embed, mask = mask) # batch, 1024
        attention_SOS = attention_SOS.unsqueeze(0) # 1, batch, 1024
        attention_SOS = attention_SOS.repeat(self.num_layers_RNN, 1, 1) # num_layers, batch, 1024
        
        start_embeds = start_embed.unsqueeze(0) # 1, batch, 1024

        # Init the hidden state
        h0 = attention_SOS
        c0 = torch.zeros_like(h0)
        
        
        pred = attention_SOS[-1].unsqueeze(0) # seq, batch, 1024
        firts_pred = torch.cat((pred, start_embeds), dim=2) # seq, batch, 2048
        pred = self.proj(firts_pred) # seq, batch, NUM_WORDS
        pred = pred.permute(1, 0, 2) # batch, seq, NUM_WORDS
 
        out = start_embeds
        
        for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
            out, (h0, c0) = self.RNN(out, (h0, c0)) # 1, batch, 1024
            
            # Cross attention
            context = self.cross_attention(roi_features, h0[-1], mask = mask) # batch, 768

            context = context.unsqueeze(0) # 1, batch, 768
            out = torch.cat((context, out), dim = 2) # seq, batch, 1024 * 2
            out = self.proj(out) # seq, batch, NUM_WORDS
            out = out.permute(1, 0, 2) # batch, seq, NUM_WORDS
            
            
            pred = torch.cat((pred, out), dim=1) # batch, seq, NUM_WORDS

            out = out.permute(1, 0, 2) # seq, batch, NUM_WORDS
            _, out = out.max(2) # seq, batch
            out = self.embed(out) # seq, batch, 512
            
        pred = pred.permute(0, 2, 1) # batch, NUM_WORDS, seq
        return pred
    
class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Make the output of the encoder consistent with the decoder
        self.q = nn.Linear(hidden_size, hidden_size) # 768 a 768
        self.k = nn.Linear(hidden_size, hidden_size) # 768 a 768
        self.v = nn.Linear(hidden_size, hidden_size) # 768 a 768

    def scoringDot(self, keys, query, mask):
        # keys: batch, 1000, 1024
        # query: batch, 1024
        query = query.unsqueeze(2) # batch, 1024, 1
        result = torch.bmm(keys, query) # batch, 1000, 1
        result = result.squeeze(2) # batch, 1000
        
        print("prev", result)
        print("mask", mask)
        result = result + mask
        print("after", result)
        
        weights = result.softmax(1) # batch, 1000
        return weights.unsqueeze(1) # batch, 1, 1000  This represents the weights of each channel
        
    def forward(self, channel_img, last_hidden_lstm, mask = None):
        # channel_img: batch, 1000, 1024
        # output_lstm: batch, 1024
        query   = torch.tanh(self.q(last_hidden_lstm)) # batch, 1024
        keys    = torch.tanh(self.k(channel_img)) # batch, 1000, 1024
        values  = torch.tanh(self.v(channel_img)) # batch, 1000, 1024
        
        weights = self.scoringDot(keys, query, mask) # batch, 1, 1000
        
        # if mask is not None:
        #     print(mask)
        #     weights = weights + mask.unsqueeze(1) # batch, 1, 1000
        #     print(weights)
        # Multiply each channel by its weight
        context = torch.bmm(weights, values) # batch, 1, 1024
        context = context.squeeze(1) # batch, 1024
        return context
        

if __name__ == "__main__":
    import numpy as np
    torch.set_printoptions(threshold=10_000)
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
    model = mobilnet_roifeatures_crossattention(text_max_len=38, teacher_forcing_ratio=0, NUM_WORDS=NUM_WORDS, word2idx=word2idx)

    batch = 4
    regions = torch.randn(batch, 1000, 1024)
    img = torch.randn(batch, 3, 224, 224)
    res = model((regions, img))
    print(res.shape) # batch, NUM_WORDS, seq