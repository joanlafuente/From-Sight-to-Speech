import torch
from torch import nn
import pandas as pd
import random
from transformers import ResNetModel, EfficientNetModel
import torchvision

base_path = '/fhome/gia07/Image_captioning/src/Dataset/'
cap_path = f'{base_path}captions.txt'
data = pd.read_csv(cap_path)

# Unique words in the dataset
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

class transformer(nn.Module):
    def __init__(self, text_max_len, rnn_layers, word2idx, NUM_WORDS, idx2word, dropout = 0, teacher_forcing_ratio = 0):
        super().__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.cnn = ResNetModel.from_pretrained('microsoft/resnet-18').to(self.DEVICE)
        self.cnn = EfficientNetModel.from_pretrained('google/efficientnet-b0').to(self.DEVICE)
        output_dim = 1280
        self.num_encoder_layers = 6
        self.num_decoder_layers = 1
        self.transformer = nn.Transformer(d_model = output_dim, nhead = 8, num_encoder_layers = self.num_encoder_layers, num_decoder_layers = self.num_decoder_layers, dim_feedforward = output_dim*2, dropout = dropout).to(self.DEVICE)
        # d_model: The number of expected features in the encoder/decoder inputs (default=512).
        # nhead: The number of heads in the multiheadattention models (default=8).
        # num_encoder_layers: The number of sub-encoder-layers in the encoder (default=6).
        # num_decoder_layers: The number of sub-decoder-layers in the decoder (default=6).
        # dim_feedforward: The dimension of the feedforward network model before the final linear layer. (default=2048).
        # dropout: The dropout value (default=0).

        self.embed = nn.Embedding(NUM_WORDS, output_dim).to(self.DEVICE)
        self.rnn = nn.LSTM(output_dim, output_dim, num_layers = 1).to(self.DEVICE)
        self.proj = nn.Linear(output_dim, NUM_WORDS).to(self.DEVICE)
        self.rnn_layers = rnn_layers
        self.TEXT_MAX_LEN = text_max_len
        self.word2idx = word2idx

    def unfreeze_params(self, value = False):
        for param in self.cnn.parameters():
            param.requires_grad = value

    def forward(self, img, ground_truth = None):
        batch_size = img.shape[0]
        feat = self.cnn(img.to(self.DEVICE)) # batch, cnn_output_dim, 1, 1
        
        img_channels = feat.last_hidden_state.to(self.DEVICE) # batch, cnn_output_dim, 7, 7
        
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, cnn_output_dim
        feat = feat.repeat(self.rnn_layers, 1, 1) # num_layers, batch, cnn_output_dim

        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        h0 = feat.repeat(self.rnn_layers, 1, 1)
        c0 = feat.repeat(self.rnn_layers, 1, 1)
        pred = inp
        for i in range(self.TEXT_MAX_LEN - 1): # rm <SOS>
            out, (h0, c0) = self.rnn(inp, (h0, c0)) # 1, batch, 512
            
            output_transformer = self.transformer(feat.repeat(self.num_encoder_layers, 1, 1), out.repeat(self.num_decoder_layers, 1, 1))
            # print(output.shape) # [num_decoder_layers, batch, out_cnn]

            pred = torch.cat((pred, output_transformer), dim=0)
            output_transformer = output_transformer.permute(1, 0, 2) # batch, num_decoder_layers, out_cnn
            output_transformer = self.proj(output_transformer) # batch, num_decoder_layers, NUM_WORDS
            output_transformer = output_transformer.permute(1, 0, 2) # num_decoder_layers, batch, NUM_WORDS
            _, output_transformer = output_transformer.max(2) # num_decoder_layers, batch

            
            out = out.permute(1, 0, 2) # batch, seq, 512
            out = self.proj(out) # batch, seq,  NUM_WORDS
            out = out.permute(1, 0, 2) # seq, batch, NUM_WORDS
            _, out = out.max(2) # seq, batch
            
            out = self.embed(out) # seq, batch, 512

            inp = out
            
        res = pred.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res) # batch, seq, 80 
        res = res.permute(0, 2, 1) # batch, 80, seq
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
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + nn.Parameter(self.pe[:,:seq_len], requires_grad=False).to(self.device)
        return x

class Transformer_positional_encoding_not_learned(nn.Module):
    def __init__(self, num_classes, depth_transformer, heads_transformer, dim_fc_transformer, drop=0.1):
        super(Transformer_positional_encoding_not_learned, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        full_cnn = torchvision.models.convnext_tiny(weights="DEFAULT")
        
        # Removing the average pooling and fully conected of the model
        modules=list(full_cnn.children())[:-2]
        self.feature_extractor=nn.Sequential(*modules)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True # Setting the CNN to be trainable, this way we are finetuning it for the specific task
         
        self.out_channels_cnn = 768 # output cnn (number of channels)
        self.dim_cnn_features = 49 # (7x7) Size of the feature maps flattened
    
        self.dim_text_features = 300 # Dimension of the text features
        self.dim = 360 # Dimension in which the images and text features are embedded

        # Linear layer to embed the text and images into the same space
        self.cnn_features_embed = nn.Linear(self.dim_cnn_features, self.dim)
        self.text_features_embed = nn.Linear(self.dim_text_features, self.dim)

        # Positional embedding for the image features and CLS token, in this case is constant (Not learned)
        self.pos_embedding = PositionalEncoder(self.dim, self.out_channels_cnn + 1, self.device)

        # CLS token, its a learnable parameter. So the model learns it during training
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=heads_transformer, dim_feedforward=dim_fc_transformer, batch_first=True, dropout=drop)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth_transformer)

        # Classification MLP, that will get the output of the transformer in the CLS token to classify it
        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(self.dim, dim_fc_transformer),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Linear(dim_fc_transformer, num_classes)
        )

    def forward(self, img, txt, text_mask):
        batch_size = img.shape[0] # We get the batch size passed to the model

        # Extract the features from the images
        image_features = self.feature_extractor(img) # Shape (batch_size, 768, 7, 7)
        # Flatten the feature maps and permute the dimensions to get the right shape for the embedding
        image_features = image_features.reshape(batch_size, self.dim_cnn_features, self.out_channels_cnn).permute(0, 2, 1) # Shape (batch_size, 49, 768)
        # Projecting the feature maps into the same space as the text features
        image_features = self.cnn_features_embed(image_features)  # Shape (batch_size, self.dim, 768)

        # We add the CLS token for the transformer at the first position
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Shape (batch_size, self.dim, 1)
        x = torch.cat((cls_tokens, image_features), dim=1) # Shape (batch_size, self.dim, 769)
        # We add the positional embedding to the image features and CLS token
        x = self.pos_embedding(x)

        # Projecting the text features into the same space as the image features
        text_features = self.text_features_embed(txt.float())
        x = torch.cat((x, text_features), dim=1) # Shape (batch_size, self.dim, 769 + max_num_words)

        # Create a mask of zeros for the image features and CLS token, so all are taken into account by the transformer
        tmp_mask = torch.zeros((batch_size, 1+self.out_channels_cnn), dtype=torch.bool).to(self.device)
        mask = torch.cat((tmp_mask, text_mask), dim=1)

        # Pass the features and mask through the transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Getting the output of the transformer encoder for the CLS token and passing it through a MLP to have the dimension
        # equal to the number of clases and be able to classify
        x = x[:, 0, :]
        x = self.fc(x)
        return x