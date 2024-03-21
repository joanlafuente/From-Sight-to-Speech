import torch
from torch import nn
from torchvision.models import convnext_tiny
from torchvision.models.feature_extraction import create_feature_extractor
import pandas as pd
import math

class Regions_transformer_updated_pretrained_linear(nn.Module):
    def __init__(self, text_max_len, teacher_forcing_ratio, NUM_WORDS, word2idx, dropout = 0.1, transformer_heads=8, transformer_layers = 2, return_attn = False, idx2word = None, pretrained_embedding = False):
        super().__init__()
        dim_embedding = 300
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
                                      
        self.linear_enc = nn.Linear(1280, dim_embedding)                 

        # Projection transformer embedds to NUM_WORDS
        self.proj = nn.Linear(dim_embedding, NUM_WORDS)
        self.relu = nn.ReLU()
        
        # Embedding layer words
        
        import gensim
        print("Loading pretrained embeddings")
        pretrained_embed = gensim.models.KeyedVectors.load_word2vec_format('/fhome/gia07/Image_captioning/src/Dataset/GoogleNews-vectors-negative300.bin', binary=True)
        print("Pretrained embeddings loaded")
        if idx2word is None:
            idx2word = {v: k for k, v in word2idx.items()}
        counter = 0
        embeddings = torch.randn(NUM_WORDS, 300)
        for i in range(NUM_WORDS):
            word = idx2word[i]
            if word in pretrained_embed:
                embeddings[word2idx[word], :] = torch.from_numpy(pretrained_embed[word].copy())
                counter += 1
        print(f"Number of words with pretrained embeddings: {counter}")
        print(f"Number of words without pretrained embeddings: {NUM_WORDS - counter}")

        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.linear_embed = nn.Linear(300, dim_embedding)

        self.word2idx = word2idx
        self.TOTAL_MAX_WORDS = text_max_len - 1  # sos token

        # Init transformer decoder
        transformer_layer = nn.TransformerDecoderLayer(d_model=dim_embedding, nhead=transformer_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(transformer_layer, num_layers=transformer_layers)
        
        # Positional encoder
        # self.positional_encoder_img = PositionalEncoder(dim_embedding, 197, self.DEVICE)
        self.positional_encoder = PositionalEncoder(dim_embedding, text_max_len, self.DEVICE)
        self.return_attn = return_attn


        if (teacher_forcing_ratio != 0) and (teacher_forcing_ratio != 1):
            raise NotImplementedError(f"Teacher forcing ratio must be 0 or 1, not {teacher_forcing_ratio}")
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def generate_mask(self, regions):
        mask = regions.sum(dim=2) == 0
        mask = mask.permute(1, 0) # batch, num_regions
        # mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and sequence length dimensions
        return mask.to(regions.device)
    
    
    def unfreeze_params(self, value = False):
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = value
            
        for param in self.embed.parameters():
            param.requires_grad = value

    def forward(self, img, regions, ground_truth = None):
        batch_size = img.shape[0]
    
        img_enc = regions.permute(1, 0, 2).to(self.DEVICE)
        mask = self.generate_mask(regions)

        img_enc = self.linear_enc(img_enc) # 197, Batch, dim_embedding
        img_enc = self.relu(img_enc)

        print(img_enc.shape)

        # img_enc = self.positional_encoder_img(img_enc) # 197, Batch, dim_embedding

        start = torch.tensor(self.word2idx['<SOS>']).to(self.DEVICE)
        start_embed = self.embed(start) # 300
        start_embed = self.linear_embed(start_embed) # dim_embedding
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, dim_embedding
        inp = start_embeds
        pad_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.DEVICE) # batch, seq

        pred = None

        if (self.teacher_forcing_ratio == 0) or (ground_truth is None): 
            # pred = inp     
            
            # to visualize attention over 1 image
            if self.return_attn:
                input_words = [[start.item()]]
            for i in range(self.TOTAL_MAX_WORDS+1):
                mask = torch.triu(torch.full((i+1, i+1), float('-inf'), device=self.DEVICE), diagonal=1)
                out = self.transformer_decoder(self.positional_encoder(inp), img_enc, tgt_mask=mask, tgt_key_padding_mask=pad_mask) # seq, batch, dim_embedding
                out = self.proj(out.permute(1, 0, 2)).permute(1, 0, 2) 
                if pred is None:
                    pred = out.permute(1, 2, 0)[:, :, -1:]
                else:
                    # print(pred.shape, out.permute(1, 2, 0)[:, :, -1:].shape)
                    pred = torch.cat((pred, out.permute(1, 2, 0)[:, :, -1:]), dim=2) # seq, batch, NUM_WORDS
                
                _, out = torch.max(out, dim=2) 
                
                pad_mask = (out == self.word2idx['<PAD>']).permute(1, 0) # batch, seq
                pad_mask = torch.cat((torch.zeros((batch_size, 1), dtype=torch.bool, device=self.DEVICE), pad_mask), dim=1) # batch, seq
                
                if self.return_attn:
                    # print(out.shape)
                    input_words.append(torch.cat((start.unsqueeze(0), out[:, 0]), dim=0).tolist())
                
                # print(pad_mask)
                out = self.embed(out)
                out = self.linear_embed(out)
                inp = torch.cat((start_embeds, out))
        else:
            pad_idx = self.word2idx['<PAD>']
            pad_mask = (ground_truth == pad_idx) # batch, seq
            pad_mask = torch.cat((torch.zeros((batch_size, 1), dtype=torch.bool, device=self.DEVICE), pad_mask), dim=1) # batch, seq
            ground_truth = self.embed(ground_truth) # batch, seq, dim_embedding
            ground_truth = self.linear_embed(ground_truth) # batch, seq, dim_embedding
            ground_truth = ground_truth.permute(1, 0, 2) # seq, batch, dim_embedding
            ground_truth = torch.cat((start_embeds, ground_truth), dim=0) # seq, batch, dim_embedding
            ground_truth = self.positional_encoder(ground_truth) # seq, batch, dim_embedding
            
            mask = torch.triu(torch.full((self.TOTAL_MAX_WORDS+1, self.TOTAL_MAX_WORDS+1), float('-inf'), device=self.DEVICE), diagonal=1)
        
            pred = self.transformer_decoder(ground_truth, img_enc, tgt_mask=mask, tgt_key_padding_mask=pad_mask) # seq, batch, dim_embedding
            
            # save_output = SaveOutput()
            # for module in self.transformer_decoder.modules():
            #     if isinstance(module, nn.MultiheadAttention):
            #         self.patch_attention(module)
            #         module.register_forward_hook(save_output)
            
            pred = pred.permute(1, 0, 2) # batch, seq, dim_embedding
            pred = self.proj(pred) # batch, seq,  NUM_WORDS
            pred = pred.permute(0, 2, 1) # batch, NUM_WORDS, seq

        if self.return_attn:
            return pred[:, :, :-1], input_words
        else:
            return pred[:, :, :-1]

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len, device):
        super().__init__()
        self.d_model = d_model
        self.device = device
        # create constant 'pe' matrix
        pe = torch.zeros(max_seq_len+1, d_model)
        for pos in range(max_seq_len+1):
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


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap

if __name__ == "__main__":
    import numpy as np
    base_path = "/fhome/gia07/Image_captioning/src/Dataset/"
    img_path = f'{base_path}Images/'
    cap_path = f'{base_path}captions_corrected.txt'
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
        # split per spaces or punctuation
        
        #remove !, (, ), ", ', :, ;, ?
        caption = caption.replace("?", "")
        caption = caption.replace("!", "")
        caption = caption.replace("(", "")
        caption = caption.replace(")", "")
        caption = caption.replace('"', "")
        caption = caption.replace("'", "")
        caption = caption.replace(":", "")
        caption = caption.replace(";", "")
        caption = caption.replace(",", "")
        caption = caption.replace("  ", " ")
        
        caption = caption.replace("-", " - ")
        caption = caption.split()
        
        # remove any word with a number
        caption = [word for word in caption if not any(char.isdigit() for char in word)]
        
        list_words.append(caption)
        unique_words.update(caption)


    # remove all words that are not in the embedding
    
    import gensim
    print("Loading pretrained embeddings")
    pretrained_embed = gensim.models.KeyedVectors.load_word2vec_format('/fhome/gia07/Image_captioning/src/Dataset/GoogleNews-vectors-negative300.bin', binary=True)
    print("Pretrained embeddings loaded")
    
    print(len(unique_words))
    unique_words = [word for word in unique_words if word in pretrained_embed]
    print(len(unique_words))
    
    # Count the number of times that each word appears in the dataset
    word_count = {}
    from collections import Counter
    word_count = Counter([word for caption in list_words for word in caption])
    # total/(num_clases * count_word)

    total = sum(word_count.values())
    num_classes = len(unique_words)

    unique_words = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + sorted(list(unique_words))
    NUM_WORDS = len(unique_words)
    idx2word = {k: v for k, v in enumerate(unique_words)}
    word2idx = {v: k for k, v in enumerate(unique_words)}
    
    if idx2word is None:
        idx2word = {v: k for k, v in word2idx.items()}
    counter = 0
    embeddings = torch.randn(len(unique_words), 300)
    for i in range(len(unique_words)):
        word = idx2word[i]
        if word in pretrained_embed:
            embeddings[word2idx[word], :] = torch.from_numpy(pretrained_embed[word].copy())
            counter += 1
        else: 
            print(word)
            
    embeddings[word2idx['<UNK>'], :] = torch.zeros(300)
    #set sos token to a tensor of zeros except for the first element which is 1
    embeddings[word2idx['<SOS>'], :] = torch.zeros(300)
    embeddings[word2idx['<SOS>'], 0] = 1
    
    embeddings[word2idx['<EOS>'], :] = torch.zeros(300)
    embeddings[word2idx['<EOS>'], 1] = 1
    
    embeddings[word2idx['<PAD>'], :] = torch.zeros(300)
    embeddings[word2idx['<PAD>'], 2] = 1
    print(f"Number of words with pretrained embeddings: {counter}")
    print(f"Number of words without pretrained embeddings: {NUM_WORDS - counter} setting them to unk")
    

    


    # Init model
    model = VIT_transformer_updated_pretrained(text_max_len=38, teacher_forcing_ratio=0, NUM_WORDS=NUM_WORDS, word2idx=word2idx, return_attn=True, transformer_heads=4)

    save_output = SaveOutput()
    patch_attention(model.transformer_decoder.layers[-1].self_attn)
    hook_handle = model.transformer_decoder.layers[-1].self_attn.register_forward_hook(save_output)

    batch_size = 3

    img = torch.randn(batch_size, 3, 224, 224)
    caption = torch.randint(0, NUM_WORDS, (batch_size, 38))    
    res, input_words = model(img, caption)
    
    # res_words = res.argmax(dim=1)
    # res_words = ["<SOS>"]+[idx2word[idx.item()] for idx in res_words[0]]
    # print(res_words.shape)
    
    print(res.shape) # batch, NUM_WORDS, seq
    
    # # print(save_output.outputs[0].shape) # batch, heads, seq, seq
    # # torch.set_printoptions(threshold=10_000)
    # # print(save_output.outputs[0][0, 0, :, :])    
    
    # # import sys
    # # sys.path.append('/fhome/gia07/Image_captioning/src/visualization_utils')
    # # # print(sys.path)
    # # from model_view import model_view
    # # model_view(save_output.outputs, tokens=res_words, html_action="save", save_path="/fhome/gia07/Image_captioning/src/visualization_utils/attention.html")
    
    
        
    # # src = ['array' , 'of', 'source', 'words']
    # # len(src) == n
    # # translation = ['array', 'of', 'translated', 'words']
    # # len(translation) == m
    # # attn = np.array shape (m,n)
    # # Construct a 2-D array of information
    # # import wandb

    # # wandb.init(project="Image_Captioning2", name="borrar")


    # # def upload_self_attention(save_output, input_words, idx2word):
    # #     for i, (src, translation) in enumerate(zip(input_words, input_words[1:])):
    # #         # print(save_output.outputs[i].shape)
    # #         attn = save_output.outputs[i][0, 0, :, :]

    # #         # print(translation)
    # #         src = [idx2word[idx] for idx in src]
    # #         translation = [idx2word[idx] for idx in translation[1:]]


    # #         attn_data = []
    # #         for m in range(attn.shape[0]):
    # #             for n in range(attn.shape[1]):
    # #                 attn_data.append([n, m, src[n], translation[m], attn[m, n]])
    # #         data_table = wandb.Table(data=attn_data, columns=["s_ind", "t_ind", "s_word", "t_word", "attn"])
    # #         fields = {
    # #             "s_index": "s_ind",
    # #             "t_index": "t_ind",
    # #             "sword": "s_word",
    # #             "tword": "t_word",
    # #             "attn": "attn"
    # #         }
    # #         wandb.log({
    # #             f"my_nlp_viz_id_{i}": wandb.plot_table(
    # #                             vega_spec_name="kylegoyette/nlp-attention-visualization",
    # #                             data_table=data_table,
    # #                             fields=fields
    # #                             )
    # #         })
        
    # # wandb.finish()

