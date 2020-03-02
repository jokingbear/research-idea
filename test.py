import numpy as np
import torch

from torchtext import vocab

columns = np.load("columns.npy")
columns[columns == "Cardiomegaly"] = "abnormal heart size"
columns[columns == "Enlarged Cardiomediastinum"] = "enlarged heart mediastinum"
columns[columns == "Pneumonia/infection"] = "pneumonia infection"

embedder = vocab.GloVe(name="6B")

embeddings = [embedder.get_vecs_by_tokens(c.split(" "), lower_case_backup=True).mean(dim=0) for c in columns]
embeddings = torch.stack(embeddings, dim=0)

np.save("embeddings", embeddings.numpy())
