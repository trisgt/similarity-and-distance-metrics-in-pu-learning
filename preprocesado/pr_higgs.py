import pandas as pd
from config.paths_config import DATASETS, ENGINEERED_DATASETS, SPLIT_DATASETS
from preprocesado.kfold import separar_dataset_en_k, generar_folds_pu

# Ruta del dataset original:
RUTA_DT = DATASETS / "higgs" / "HIGGS.csv"

# Ruta del dataset tras separación en K-folds:
RUTA_FOLDS = SPLIT_DATASETS / "higgs"

# Ruta de los folds tras convertirlos en PU:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "higgs"

# Parámetros para el split del dataset en k-folds:
NUM_FOLDS = 5
SEMILLA_SPLIT = 1

# Parámetros para el guardado de los datasets:
GUARDADO_NPY = True # Si es falso, los datasets se guardan como ".csv"

# Parámetros para la conversión a PU:
CL_POSITIVAS = 1
PORCENTAJE_POS = 0.6
SEMILLA_PR = 1

# Función para cargar el dataset original:
def cargar_higgs(path_data):
    dataframe = pd.read_csv(path_data, sep = ",", header = None)

    # "y" es la primera columna. Como Higgs no tiene cabecera, se busca sin un nombre:
    y = dataframe.iloc[:, 0].astype(int)

    # "X" son todas las demas columnas:
    X = dataframe.drop(columns = dataframe.columns[0])

    return X, y

# Función principal:
if __name__ == "__main__":
    X, y = cargar_higgs(RUTA_DT) # Carga del dataset original

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