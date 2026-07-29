import pandas as pd
from config.paths_config import DATASETS, ENGINEERED_DATASETS, SPLIT_DATASETS
from preprocesado.kfold import separar_dataset_en_k, generar_folds_pu

# Rutas del dataset original de entrenamiento y de test:
RUTA_DT_TRAIN = DATASETS / "isolet" / "isolet1+2+3+4.data"
RUTA_DT_TEST = DATASETS / "isolet" / "isolet5.data"

# Ruta del dataset tras separación en K-folds:
RUTA_FOLDS = SPLIT_DATASETS / "isolet"

# Ruta de los folds tras convertirlos en PU:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "isolet"

# Parámetros para el split del dataset en k-folds:
NUM_FOLDS = 5 # Por defecto, 5 (5-fold)
SEMILLA_SPLIT = 1

# Parámetros para el preprocesado:
CL_POSITIVAS = 1
PORCENTAJE_POS = 0.6
SEMILLA_PR = 1

# Función para cargar el dataset original. Se concatenan los de entrenamiento y test para luego hacer k-folds:
def cargar_isolet(path_train, path_test):
    dataframe_train = pd.read_csv(path_train, sep = ",", header = None)
    dataframe_test = pd.read_csv(path_test, sep = ",", header = None)

    # Tras leer el dataset de entrenamiento y test, ambos se concatenan:
    dataframe = pd.concat([dataframe_train, dataframe_test], ignore_index = True)

    # "y" es la última columna. Como ISOLET no tiene cabecera, se busca sin un nombre:
    y = dataframe.iloc[:, -1].astype(int)

    # "X" son todas las demas columnas:
    X = dataframe.drop(columns = dataframe.columns[-1])

    return X, y

# Función principal:
if __name__ == "__main__":
    # Carga y concatenación de los dataset originales:
    X, y = cargar_isolet(RUTA_DT_TRAIN, RUTA_DT_TEST)

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