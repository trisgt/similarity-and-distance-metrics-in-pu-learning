import pandas as pd
from config.paths_config import DATASETS, ENGINEERED_DATASETS, SPLIT_DATASETS

try:
    from preprocesado.preprocesado_dt import convertir_a_pu, guardar_dataset_pu
    from preprocesado.kfold import separar_dataset_en_k, generar_folds_pu
except ModuleNotFoundError:
    from preprocesado_dt import convertir_a_pu, guardar_dataset_pu
    from kfold import separar_dataset_en_k, generar_folds_pu

# Ruta del dataset original:
RUTA_DT = DATASETS / "bank+marketing" / "bank-additional" / "bank-additional" / "bank-additional-full.csv"

# Ruta del dataset tras separación en K-folds:
RUTA_FOLDS = SPLIT_DATASETS / "bank_marketing"

# Ruta de los folds tras convertirlos en PU:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "bank_marketing"

# Parámetros para el split del dataset en k-folds:
NUM_FOLDS = 5 # Por defecto, 5 (5-fold)
SEMILLA_SPLIT = 1

# Parámetros para la conversión a PU:
CL_POSITIVAS = 1
PORCENTAJE_POS = 0.6
SEMILLA_PR = 1

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