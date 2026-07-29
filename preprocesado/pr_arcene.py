import pandas as pd
from config.paths_config import DATASETS, ENGINEERED_DATASETS, SPLIT_DATASETS
from preprocesado.kfold import separar_dataset_en_k, generar_folds_pu

# Rutas del dataset original de entrenamiento y de test:
RUTA_DT_TRAIN = DATASETS / "arcene" / "ARCENE" / "arcene_train.data"
RUTA_DT_TRAIN_LABELS = DATASETS / "arcene" / "ARCENE" / "arcene_train.labels"
RUTA_DT_VALID = DATASETS / "arcene" / "ARCENE" / "arcene_valid.data"
RUTA_DT_VALID_LABELS = DATASETS / "arcene" / "arcene_valid.labels"

# Ruta del dataset tras separación en K-folds:
RUTA_FOLDS = SPLIT_DATASETS / "arcene"

# Ruta de los folds tras convertirlos en PU:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "arcene"

# Parámetros para el split del dataset en k-folds:
NUM_FOLDS = 5 # Por defecto, 5 (5-fold)
SEMILLA_SPLIT = 1

# Parámetros para el preprocesado:
CL_POSITIVAS = 1
PORCENTAJE_POS = 0.6
SEMILLA_PR = 1

# Función para cargar el dataset original. Se concatenan los de entrenamiento y test para luego hacer k-folds:
def cargar_arcene(path_train, path_train_labels, path_valid, path_valid_labels):
    # Como los conjuntos y sus labels ("y") están separados, hay que cargar 4 conjuntos:
    dataframe_train = pd.read_csv(path_train, sep = r"\s+", header = None)
    dataframe_valid = pd.read_csv(path_valid, sep = r"\s+", header = None)
    dataframe_train_labels = pd.read_csv(path_train_labels, header = None).squeeze("columns")
    dataframe_valid_labels = pd.read_csv(path_valid_labels, header = None).squeeze("columns")

    # Tras leer el dataset de entrenamiento y test, ambos se concatenan, "X" e "y" por separado:
    X = pd.concat([dataframe_train, dataframe_valid], ignore_index = True)
    y = pd.concat([dataframe_train_labels, dataframe_valid_labels], ignore_index = True)

    # "y" se convierte a {0, 1} en vez de {-1, 1}:
    y = (y == 1).astype(int)

    return X, y

# Función principal:
if __name__ == "__main__":
    # Carga y concatenación de los dataset originales:
    X, y = cargar_arcene(RUTA_DT_TRAIN, RUTA_DT_TRAIN_LABELS, RUTA_DT_VALID, RUTA_DT_VALID_LABELS)

    # Separación en K-folds:
    separar_dataset_en_k(
        X,
        y,
        NUM_FOLDS,
        RUTA_FOLDS,
        SEMILLA_SPLIT
    )

    # Transformación de los folds a PU. La conversión se hace dentro de
    # la propia función "generar_folds_pu":
    generar_folds_pu(
        RUTA_FOLDS,
        RUTA_FOLDS_PU,
        NUM_FOLDS,
        CL_POSITIVAS,
        PORCENTAJE_POS,
        SEMILLA_PR,
        False,
        True
    )

    print("Preprocesado completado")