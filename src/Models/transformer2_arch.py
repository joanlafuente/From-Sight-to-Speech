import torch
from torch import nn
import pandas as pd
import random
from transformers import ResNetModel, EfficientNetModel
import torchvision

class transformer2(nn.Module):
    def __init__(self, text_max_len, word2idx, NUM_WORDS, idx2word, dropout = 0, teacher_forcing_ratio = 0):
        super().__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.cnn = ResNetModel.from_pretrained('microsoft/resnet-18').to(self.DEVICE)
        self.cnn = EfficientNetModel.from_pretrained('google/efficientnet-b0').to(self.DEVICE)
        output_dim = NUM_WORDS
        self.num_encoder_layers = 6
        self.num_decoder_layers = text_max_len
        print(NUM_WORDS)
        self.transformer = nn.Transformer(d_model = output_dim, nhead = 4, 
                                          num_encoder_layers = self.num_encoder_layers, 
                                          num_decoder_layers = self.num_decoder_layers, 
                                          dim_feedforward = output_dim*2, 
                                          dropout = dropout).to(self.DEVICE)

        # self.embed = nn.Embedding(NUM_WORDS, output_dim).to(self.DEVICE)
        self.proj = nn.Linear(output_dim, NUM_WORDS).to(self.DEVICE)
        self.TEXT_MAX_LEN = text_max_len
        self.word2idx = word2idx

    def unfreeze_params(self, value = False):
        for param in self.cnn.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth = None):
        print(img.shape)
        batch_size = img.shape[0]
        feat = self.cnn(img.to(self.DEVICE)) # batch, cnn_output_dim, 1, 1
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, cnn_output_dim

        output_transformer = self.transformer(feat.repeat(self.num_encoder_layers, 1, 1), feat.repeat(self.num_decoder_layers, 1, 1))

        # output_transformer has shape [num_words_sentence, batch, size_vocab]
        print(output_transformer.shape)
        # output_transformer = output_transformer.permute(1, 0, 2)
        # output_transformer = self.proj(output_transformer)
        # output_transformer = output_transformer.permute(1, 0, 2)
        res = output_transformer.argmax(2) # num_words_sentence, batch
        return res
        