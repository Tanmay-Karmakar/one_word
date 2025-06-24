import numpy as np
import sys

embedding_path = sys.argv[1]
save_embed_npy_path = './models/Luyu-bert-base-mdoc-bm25-npymat'

embeddings = [] 
with open(embedding_path, 'r') as ifile:
    for line in ifile:
        embedding = [float(num) for num in line.strip().split()[1:]]
        embeddings.append(embedding)
embeddings = np.array(embeddings)
print(embeddings.T.shape)
norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = np.asarray(embeddings / norm, "float32")

product = np.dot(embeddings, embeddings.T)
print(product.shape,'\n\n')
print(product)
np.save(save_embed_npy_path, product)