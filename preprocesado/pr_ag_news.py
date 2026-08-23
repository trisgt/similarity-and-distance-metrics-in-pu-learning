import pandas as pd
from config.paths_config import DATASETS, ENGINEERED_DATASETS, SPLIT_DATASETS
from preprocesado.pr_textos import generar_embeddings
from preprocesado.kfold import separar_dataset_en_k, generar_folds_pu

# Rutas del dataset original de entrenamiento y de test:
RUTA_DT_TRAIN = DATASETS / "ag_news" / "train.csv"
RUTA_DT_TEST = DATASETS / "ag_news" / "test.csv"

# Ruta del dataset tras separación en K-folds:
RUTA_FOLDS = SPLIT_DATASETS / "ag_news"

# Ruta de los folds tras convertirlos en PU:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "ag_news"

# Parámetros para el split del dataset en k-folds:
NUM_FOLDS = 5
SEMILLA_SPLIT = 1

# Parámetros para el guardado de los datasets:
GUARDADO_NPY = True # Si es falso, los datasets se guardan como ".csv"

# Parámetros para la conversión a PU:
CL_POSITIVAS = 1
PORCENTAJE_POS = 0.6
SEMILLA_PR = 1

# Parámetros para los embeddings del texto:
BATCH_SIZE = 32
DEVICE = ["cpu", "cpu", "cpu", "cpu"]

# Función para cargar el dataset original:
def cargar_ag_news(path_train, path_test):
    dataframe_train = pd.read_csv(path_train, sep = ",")
    dataframe_test = pd.read_csv(path_test, sep = ",")

    # Tras leer el dataset de entrenamiento y test, ambos se concatenan:
    dataframe = pd.concat([dataframe_train, dataframe_test], ignore_index = True)

    # Se concatenan título y descripción para obtener los artículos:
    news_articles = (
        dataframe["Title"].astype(str) +
        ". " +
        dataframe["Description"].astype(str)
    ).tolist()

    # Utilizando los artículos, generamos los embeddings ("X"):
    X = generar_embeddings(news_articles, BATCH_SIZE, DEVICE)

    # Para "y", aprovechamos que los índices de la clase aparecen en formato numérico.
    # Las clases originales son 1-4, asi que las convertimos a 0-3:
    y = dataframe["Class Index"].to_numpy() - 1

    return X, y

# Función principal:
if __name__ == "__main__":
    X, y = cargar_ag_news(RUTA_DT_TRAIN, RUTA_DT_TEST) # Carga del dataset original

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