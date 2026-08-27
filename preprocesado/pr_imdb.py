import pandas as pd
from config.paths_config import DATASETS, ENGINEERED_DATASETS, SPLIT_DATASETS
from config.preproc_config import *
from preprocesado.pr_textos import generar_embeddings
from preprocesado.kfold import separar_dataset_en_k, generar_folds_pu

# Ruta del dataset original:
RUTA_DT = DATASETS / "imdb" / "IMDB Dataset.csv"

# Ruta del dataset tras separación en K-folds:
RUTA_FOLDS = SPLIT_DATASETS / "imdb"

# Ruta de los folds tras convertirlos en PU:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "imdb"

# Parámetros para el guardado de los datasets:
GUARDADO_NPY = True # Si es falso, los datasets se guardan como ".csv"

# Parámetros para la conversión a PU:
CL_POSITIVAS = 1 # Clases consideradas como positivas

# Función para cargar el dataset original:
def cargar_imdb(path_data):
    dataframe = pd.read_csv(path_data, sep = ",")

    # Obtenemos las reseñas:
    reviews = dataframe["review"].astype(str).tolist()

    # Utilizando las reseñas, generamos los embeddings ("X"):
    X = generar_embeddings(reviews, BATCH_SIZE, DEVICE)

    # Para "y", convertimos el sentimiento ("positive"/"negative") a una etiqueta binaria (1/0):
    y = (dataframe["sentiment"] == "positive").astype(int).to_numpy()

    return X, y

# Función principal:
if __name__ == "__main__":
    X, y = cargar_imdb(RUTA_DT) # Carga del dataset original

    # Separación en K-folds:
    separar_dataset_en_k(
        X,
        y,
        NUM_FOLDS,
        RUTA_FOLDS,
        GUARDADO_NPY,
        SEMILLA_SPLIT
    )

    # Transformación de los folds a PU. La conversión se hace dentro de
    # la propia función "generar_folds_pu":
    generar_folds_pu(
        RUTA_FOLDS,
        RUTA_FOLDS_PU,
        GUARDADO_NPY,
        NUM_FOLDS,
        CL_POSITIVAS,
        PORCENTAJE_POS,
        SEMILLA_PR
    )

    print("Preprocesado completado")