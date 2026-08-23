import numpy as np
from sentence_transformers import SentenceTransformer

# Función para generar embeddings a partir de un conjunto de textos
# utilizando un transformer. Se utiliza el modelo "all-mpnet-base-v2":
def generar_embeddings(textos, batch_size = 32, device = ["cpu"]):
    # Se carga el modelo de transformer:
    modelo = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Se generan los embeddings utilizando el modelo:
    embeddings = modelo.encode(
        textos,
        batch_size = batch_size,
        show_progress_bar = True,
        convert_to_numpy = True,
        device = device)

    return embeddings.astype(np.float32)