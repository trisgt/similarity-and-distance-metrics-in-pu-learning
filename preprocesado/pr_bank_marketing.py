import pandas as pd
import os
from sklearn.model_selection import train_test_split

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from config import DATASETS, ENGINEERED_DATASETS, SPLIT_DATASETS

try:
    from preprocesado.preprocesado_dt import convertir_a_pu, guardar_dataset
except ModuleNotFoundError:
    from preprocesado_dt import convertir_a_pu, guardar_dataset

# Ruta del dataset original:
RUTA_DT = DATASETS / "bank+marketing" / "bank-additional" / "bank-additional-full.csv"

# Ruta de los datasets tras separación en conjuntos de entrenamiento y test:
RUTA_DT_TRAIN = SPLIT_DATASETS / "bank+marketing" / "bank-additional" / "bank-additional-full-train.csv"
RUTA_DT_TEST = SPLIT_DATASETS / "bank+marketing" / "bank-additional" / "bank-additional-full-test.csv"

# Ruta del dataset convertido a PU:
RUTA_DT_PU = ENGINEERED_DATASETS / "bank_marketing"

# Parámetros para el preprocesado:
CL_POSITIVAS = 1
PORCENTAJE_POS = 0.6
SEMILLA_PR = 1

# Parámetros para la creación de conjuntos de entrenamiento y test:
PORCENTAJE_CONJ_TEST = 0.2
SEMILLA_SPLIT = 1

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

    return X, y

# Función para separar el dataset original en conjuntos de entrenamiento y test:
def separar_bank_marketing(X, y, porcentaje_test, ruta_train, ruta_test, semilla = None):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = porcentaje_test,
        random_state = semilla,
        stratify = y # Para asegurar que se conserve la misma proporción entre clases
    )

    # Se hace One-Hot Encoding de los conjuntos:
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_test = X_test.reindex(columns = X_train.columns, fill_value = 0)

    # Guardamos los conjuntos:
    guardar_split_dataset_bank_marketing(X_train, y_train, ruta_train)
    guardar_split_dataset_bank_marketing(X_test, y_test, ruta_test)

    return X_train, X_test, y_train, y_test

# Función auxiliar para guardar un dataset (train/test) como .csv:
def guardar_split_dataset_bank_marketing(X, y, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok = True)

    # Se copian los valores de "X" e "y":
    dataframe = X.copy()
    dataframe["y"] = y.values

    dataframe.to_csv(ruta, index = False, sep = ";")

# Función principal:
if __name__ == "__main__":
    X, y = cargar_bank_marketing(RUTA_DT) # Carga del dataset original

    # Separación en conjuntos de entrenamiento y test:
    X_train, X_test, y_train, y_test = separar_bank_marketing(
        X,
        y,
        PORCENTAJE_CONJ_TEST,
        RUTA_DT_TRAIN,
        RUTA_DT_TEST,
        SEMILLA_SPLIT
    )

    # Preprocesado:
    X, y, y_gt = convertir_a_pu(X_train, y_train, CL_POSITIVAS, PORCENTAJE_POS, SEMILLA_PR)
    guardar_dataset(X, y, y_gt, RUTA_DT_PU, False, True)