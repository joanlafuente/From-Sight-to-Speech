from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

import torch

def SentEmb_loss(sent1, sent2, distance = 'euclidean'):
    for i in range(len(sent1)):
        sent1[i] = sent1[i].lower()
    
    for i in range(len(sent2)):
        sent2[i] = sent2[i].lower()
    
    emb1 = model.encode([sent1])
    emb2 = model.encode([sent2])
    
    emb1 = torch.tensor(emb1)
    emb2 = torch.tensor(emb2)

    if distance == 'euclidean':
        dist = torch.nn.functional.pairwise_distance(emb1, emb2)
    
    elif distance == 'cosine':
        dist = torch.nn.functional.cosine_similarity(emb1, emb2)

    return dist