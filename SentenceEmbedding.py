from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

#Our sentences we like to encode
sentences = [
"Two young guys with shaggy hair look at their hands while hanging out in the yard .",
# "two young guys with shaggy hair look at their hands while hanging out in the yard .",
"Two young , White males are outside near many bushes .",
# "Two men in green shirts are standing in a yard .",
# "A man in a blue shirt standing in a garden .",
# "Two friends enjoy time spent together ."
]

for i in range(len(sentences)):
    sentences[i] = sentences[i].lower()

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
L_embed = []
for sentence, embedding in zip(sentences, embeddings):
    # print("Sentence:", sentence)
    # print("Embedding:", embedding)
    # print("")
    L_embed.append(embedding)

# compute the matrix of euclidean distance
import numpy as np
L_embed = np.array(L_embed)

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

dist = euclidean_distances(L_embed)
print(dist)

# compute the matrix of cosine distance
dist = cosine_distances(L_embed)
print(dist)
