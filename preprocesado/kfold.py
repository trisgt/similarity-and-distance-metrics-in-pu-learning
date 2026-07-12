import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from preprocesado.preprocesado_dt import convertir_a_pu, guardar_dataset_pu


# Función para separar un dataset en K-folds:
def separar_dataset_en_k(X, y, k, ruta_guardado, semilla = None):
    # Se divide el set en K-folds estratificados, con lo que se mantiene
    # la misma proporción de positivos y negativos en cada división, aproximadamente:
    kfold = StratifiedKFold(
        n_splits = k,
        shuffle = True,
        random_state = semilla
    )

    # Se crea el directorio donde se guardarán los folds:
    os.makedirs(ruta_guardado, exist_ok = True)

    # Bucle principal:
    fold_num = 1
    for idx in kfold.split(X, y):
        # Sólo nos interesa el fold (conj. test) de cada división:
        fold_idx = idx[1]

        # Se obtienen los índices de "X" e "y" dados por el fold:
        X_fold = X.iloc[fold_idx]
        y_fold = y.iloc[fold_idx]

        # Obtenemos la ruta de cada fold en particular como un Path:
        nombre_fold = f"fold_{fold_num}.csv"
        ruta_fold = Path(ruta_guardado) / nombre_fold

        # Guardamos el fold en la anterior ruta:
        guardar_fold(X_fold, y_fold, ruta_fold)

        fold_num += 1
    
    print(f"Se han creado {k} folds en: {ruta_guardado}")


# Función auxiliar para guardar cada uno de los folds a un .csv:
def guardar_fold(X, y, ruta):
    # Creamos la ruta del fold:
    os.makedirs(os.path.dirname(ruta), exist_ok = True)

    # Copiamos los valores de "X" e "y" a un dataframe:
    dataframe = X.copy()
    dataframe["y"] = y.values

    # Guardamos el archivo CSV:
    dataframe.to_csv(ruta, index = False, sep = ";")


# Función para generar folds PU a partir de los folds originales:
def generar_folds_pu(ruta_original, ruta_pu, k, clases_positivas, porcentaje_positivos, semilla = None, npy = False, csv = True):
    # Primero, se genera el directorio para los folds PU:
    os.makedirs(ruta_pu, exist_ok = True)

    # Bucle principal:
    for fold_num in range(1, k + 1):
        # Se obtiene la ruta de cada fold original como un Path:
        nombre_fold = f"fold_{fold_num}.csv"
        ruta_fold = Path(ruta_original) / nombre_fold

        # Se obtiene el dataframe a partir del fold, y se separa en "X" e "y":
        dataframe = pd.read_csv(ruta_fold, sep = ";")
        X = dataframe.drop(columns = "y")
        y = dataframe["y"]

        # Se convierte el fold a PU:
        X_pu, y_pu, y_gt = convertir_a_pu(X, y, clases_positivas, porcentaje_positivos, semilla)

        # Se guarda el fold en la ruta deseada:
        ruta_pu = Path(ruta_pu)
        guardar_dataset_pu(X_pu, y_pu, y_gt, ruta_pu, nombre_fold, npy, csv)
    
    print(f"Se han creado {k} folds PU en: {ruta_pu}")


# Función para juntar varios folds del mismo dataset en un solo dataset (para entrenamiento):
def juntar_folds_separados(ruta_folds, folds):
    # Creamos dos listas para "X" e "y", que luego servirán para concatenar los folds:
    lista_X = []
    lista_y = []

    # Bucle principal:
    for fold_num in folds:
        # Se obtiene la ruta de cada fold como un Path:
        nombre_fold = f"fold_{fold_num}.csv"
        ruta_fold = Path(ruta_folds) / nombre_fold

        # Se obtiene el dataframe a partir del fold, y se separa en "X" e "y" ("y_gt" es ignorado):
        dataframe = pd.read_csv(ruta_fold, sep = ";")

        # El dataframe se separa en "X" e "y". Si hubiese una columna "y_gt" (folds PU), se descarta:
        if "y_gt" in dataframe.columns:
            X = dataframe.drop(columns = ["y", "y_gt"])
        else:
            X = dataframe.drop(columns = "y") # Evitamos que de error
        y = dataframe["y"]

        # Se añaden los valores de "X" e "y" a la lista
        lista_X.append(X)
        lista_y.append(y)
    
    # Se concatenan todos los folds en un solo dataset:
    X = pd.concat(lista_X, ignore_index = True)
    y = pd.concat(lista_y, ignore_index = True)

    # El dataset no se guarda, se devuelve directamente:
    return X, y