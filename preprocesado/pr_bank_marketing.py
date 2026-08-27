import pandas as pd
from config.paths_config import DATASETS, ENGINEERED_DATASETS, SPLIT_DATASETS
from config.preproc_config import *
from preprocesado.kfold import separar_dataset_en_k, generar_folds_pu

# Ruta del dataset original:
RUTA_DT = DATASETS / "bank+marketing" / "bank-additional" / "bank-additional" / "bank-additional-full.csv"

# Ruta del dataset tras separación en K-folds:
RUTA_FOLDS = SPLIT_DATASETS / "bank_marketing"

# Ruta de los folds tras convertirlos en PU:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "bank_marketing"

# Parámetros para el guardado de los datasets:
GUARDADO_NPY = False # Si es falso, los datasets se guardan como ".csv"

# Parámetros para la conversión a PU:
CL_POSITIVAS = 1 # Clases consideradas como positivas

# Función para cargar el dataset original:
def cargar_bank_marketing(path_data):
    dataframe = pd.read_csv(path_data, sep = ";")

    # Para "y", se comprueba si el valor de la columna es "yes" o "no", convertido a True o False.
    # Si ya se había convertido previamente a números, se comprueban estos:
    if dataframe["y"].dtype == object:
        y = (dataframe["y"] == "yes").astype(int)
    else:
        y = dataframe["y"].astype(int)
    
    # Ahora eliminamos la columna "y", con lo que nos queda "X":
    X = dataframe.drop(columns = ["y"])

    # Por último, se hace One-Hot Encoding de "X":
    X = pd.get_dummies(X)

    return X, y

# Función principal:
if __name__ == "__main__":
    X, y = cargar_bank_marketing(RUTA_DT) # Carga del dataset original

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