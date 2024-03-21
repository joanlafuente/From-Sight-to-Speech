import torch
from torch import nn
from transformers import EfficientNetModel
import pandas as pd
import math

class efficientb3_transformer(nn.Module):
    def __init__(self, text_max_len, teacher_forcing_ratio, NUM_WORDS, word2idx, dropout = 0.1, transformer_heads=8, transformer_layers = 2, return_attn = False, idx2word = None):
        super().__init__()
        dim_embedding = 1536
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # CNN
        self.resnet = EfficientNetModel.from_pretrained('google/efficientnet-b3').to(self.DEVICE)

        # Projection transformer embedds to NUM_WORDS
        self.proj = nn.Linear(dim_embedding, NUM_WORDS)
        # Embedding layer words
        self.embed = nn.Embedding(NUM_WORDS, dim_embedding)

        self.word2idx = word2idx
        self.TOTAL_MAX_WORDS = text_max_len

        # Init transformer decoder
        transformer_layer = nn.TransformerDecoderLayer(d_model=dim_embedding, nhead=transformer_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(transformer_layer, num_layers=transformer_layers)
        
        # Positional encoder
        max_len = text_max_len if text_max_len > 49 else 49
        self.positional_encoder = PositionalEncoder(dim_embedding, max_len, self.DEVICE)

        self.return_attn = return_attn


        if (teacher_forcing_ratio != 0) and (teacher_forcing_ratio != 1):
            raise NotImplementedError(f"Teacher forcing ratio must be 0 or 1, not {teacher_forcing_ratio}")
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def unfreeze_params(self, value = False):
        for param in self.resnet.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth = None):
        # raise NotImplementedError("This model is not implemented completely")
    
        batch_size = img.shape[0]
    
        img_enc = self.resnet(img).last_hidden_state # batch, dim_embedding, 7, 7
        img_enc = img_enc.view(img_enc.shape[0], img_enc.shape[1], -1) # Batch, dim_embedding, 49
        img_enc = img_enc.permute(2, 0, 1) # 49, Batch, dim_embedding

        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start) # dim_embedding
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, dim_embedding
        inp = start_embeds

        if (self.teacher_forcing_ratio == 0) or (ground_truth is None):      
            for i in range(self.TOTAL_MAX_WORDS-1): # rm <SOS>
                out = self.transformer_decoder(self.positional_encoder(inp), img_enc) # 1, batch, dim_embedding
                inp = torch.cat((inp, out[-1:, :, :]), dim=0)
            pred = inp
        else:
            ground_truth = self.embed(ground_truth) # batch, seq, dim_embedding
            ground_truth = ground_truth.permute(1, 0, 2) # seq, batch, dim_embedding
            ground_truth = self.positional_encoder(ground_truth) # seq, batch, dim_embedding
            pred = self.transformer_decoder(ground_truth, img_enc) # seq, batch, dim_embedding
            # Remove last prediction as it out of the max sequence length
            pred = torch.cat((inp, pred), dim=0)[:-1, :, :]
            
        res = pred.permute(1, 0, 2) # batch, seq, dim_embedding
        res = self.proj(res) # batch, seq,  NUM_WORDS
        res = res.permute(0, 2, 1) # batch, NUM_WORDS, seq
        return res

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len, device):
        super().__init__()
        self.d_model = d_model
        self.device = device
        # create constant 'pe' matrix
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                # Get the positional encoding for each position of the sequence
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # x # seq, batch, d_model
        x = x.permute(1, 0, 2) # batch, seq, d_model
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + nn.Parameter(self.pe[:,:seq_len], requires_grad=False).to(self.device)
        x = x.permute(1, 0, 2) # seq, batch, d_model
        return x  

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
    model = efficientb3_transformer(text_max_len=38, teacher_forcing_ratio=0, NUM_WORDS=NUM_WORDS, word2idx=word2idx)

    img = torch.randn(2, 3, 224, 224)
    caption = torch.randint(0, NUM_WORDS, (2, 38))
    res = model(img, caption)
    print(res.shape) # batch, NUM_WORDS, seq
