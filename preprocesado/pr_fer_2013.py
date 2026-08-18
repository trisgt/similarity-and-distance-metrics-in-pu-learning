import numpy as np
import pandas as pd
from config.paths_config import DATASETS, ENGINEERED_DATASETS, SPLIT_DATASETS
from preprocesado.pr_imagenes import cargar_imagenes
from preprocesado.kfold import separar_dataset_en_k, generar_folds_pu

# Rutas del dataset original de entrenamiento y de test:
RUTA_DT_TRAIN = DATASETS / "fer-2013" / "train"
RUTA_DT_TEST = DATASETS / "fer-2013" / "test"

# Ruta del dataset tras separación en K-folds:
RUTA_FOLDS = SPLIT_DATASETS / "fer-2013"

# Ruta de los folds tras convertirlos en PU:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "fer-2013"


# Parámetros para el split del dataset en k-folds:
NUM_FOLDS = 5
SEMILLA_SPLIT = 1

# Parámetros para el guardado de los datasets:
GUARDADO_NPY = True # Si es falso, los datasets se guardan como ".csv"

# Parámetros para el preprocesado:
CL_POSITIVAS = [0, 1, 2]
PORCENTAJE_POS = 0.6
SEMILLA_PR = 1

# Parámetros para imágenes:
ESCALA_GRISES = True

# Función para cargar el dataset original. Se concatenan los de entrenamiento y test para luego hacer k-folds:
def cargar_fer_2013(path_train, path_test):
    # Etiquetas de las clases de FER-2013:
    clases = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6
    }

    X_train, y_train = cargar_imagenes(path_train, clases, ESCALA_GRISES)
    X_test, y_test = cargar_imagenes(path_test, clases, ESCALA_GRISES)

    # Tras leer el dataset de entrenamiento y test, ambos se concatenan:
    X = np.concatenate([X_train, X_test], axis = 0)
    y = np.concatenate([y_train, y_test], axis = 0)

    print("TEST:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y

# Función principal:
if __name__ == "__main__":
    # Carga y concatenación de los dataset originales:
    X, y = cargar_fer_2013(RUTA_DT_TRAIN, RUTA_DT_TEST)

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