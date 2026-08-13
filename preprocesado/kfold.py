import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from preprocesado.pu_engineering import convertir_a_pu, guardar_dataset_pu


# Función para separar un dataset en K-folds:
def separar_dataset_en_k(X, y, k, ruta_guardado, semilla = None, guardado_npy = False):
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
        # Sólo nos interesa el fold (conjunto de test) de cada división:
        fold_idx = idx[1]

        # Se obtienen los índices de "X" e "y" dados por el fold:
        X_fold = X.iloc[fold_idx]
        y_fold = y.iloc[fold_idx]

        # Guardamos el fold en ".npy" o en ".csv", dependiendo de la opción indicada:
        guardar_fold(X_fold, y_fold, ruta_guardado, fold_num, guardado_npy)

        fold_num += 1
    
    print(f"Se han creado {k} folds en: {ruta_guardado}")


# Función auxiliar para guardar cada uno de los folds, tanto a ".csv" como a ".npy":
def guardar_fold(X, y, ruta, fold_num, guardado_npy = False):
    # Establecemos el nombre del fold:
    nombre_fold = f"fold_{fold_num}"

    # Si la opción está activada, guardamos el archivo como ".npy":
    if guardado_npy:
        # Obtenemos la ruta como un Path, separando "X" e "y":
        ruta_X = Path(ruta) / f"{nombre_fold}_X.npy"
        ruta_y = Path(ruta) / f"{nombre_fold}_y.npy"

        # Guardamos "X" e "y" por separado con las anteriores rutas:
        np.save(ruta_X, X)
        np.save(ruta_y, y)

    # Si no, guardamos los archivos como ".csv":
    else:
        # Obtenemos la ruta como un Path:
        ruta_csv = Path(ruta) / f"{nombre_fold}.csv"

        # Copiamos los valores de "X" e "y" a un dataframe:
        dataframe = X.copy()
        dataframe["y"] = y.values

        # Guardamos el archivo .csv con la anterior ruta:
        dataframe.to_csv(ruta_csv, index = False, sep = ";")


# Función para generar folds PU a partir de los folds originales:
def generar_folds_pu(ruta_original, ruta_pu, k, clases_positivas, porcentaje_positivos, semilla = None, guardado_npy = False):
    # Primero, se genera el directorio para los folds PU:
    os.makedirs(ruta_pu, exist_ok = True)

    # Bucle principal:
    for fold_num in range(1, k + 1):
        # Establecemos el nombre del fold:
        nombre_fold = f"fold_{fold_num}"

        # Si se guardaron los folds originales como ".npy":
        if guardado_npy:
            # Se obtienen las rutas de "X" e "y" originales como un Path:
            ruta_X = Path(ruta_original) / f"{nombre_fold}_X.npy"
            ruta_y = Path(ruta_original) / f"{nombre_fold}_y.npy"

            # Se cargan los archivos:
            X = np.load(ruta_X)
            y = np.load(ruta_y)

        # Si se guardaron los folds originales como ".csv":
        else:
            # Se obtiene la ruta original como un Path:
            ruta_fold = Path(ruta_original) / f"{nombre_fold}.csv"

            # Se obtiene el dataframe a partir del fold, y se separa en "X" e "y":
            dataframe = pd.read_csv(ruta_fold, sep = ";")
            X = dataframe.drop(columns = "y")
            y = dataframe["y"]

        # Se convierte el fold a PU:
        X_pu, y_pu, y_gt = convertir_a_pu(X, y, clases_positivas, porcentaje_positivos, semilla, nombre_fold)

        # Se guarda el fold en la ruta deseada:
        ruta_pu = Path(ruta_pu)
        guardar_dataset_pu(X_pu, y_pu, y_gt, ruta_pu, nombre_fold, guardado_npy)
    
    print(f"Se han creado {k} folds PU en: {ruta_pu}")


# Función para cargar un fold de un dataset, tanto ".csv" como ".npy":
def cargar_fold(ruta_folds, fold_num, guardado_npy = False):
    # Establecemos el nombre del fold:
    nombre_fold = f"fold_{fold_num}"

    # Si el fold estaba guardado como ".npy":
    if guardado_npy:
        # Se obtienen las rutas de "X" e "y" como un Path:
        ruta_X = Path(ruta_folds) / f"{nombre_fold}_X.npy"
        ruta_y = Path(ruta_folds) / f"{nombre_fold}_y.npy"

        # Se cargan los archivos:
        X = np.load(ruta_X)
        y = np.load(ruta_y)

    # Si el fold estaba guardado como ".csv":
    else:
        # Se obtiene la ruta como un Path:
        ruta_fold = Path(ruta_folds) / f"{nombre_fold}.csv"

        # Se obtiene el dataframe del fold:
        dataframe = pd.read_csv(ruta_fold, sep = ";")

        # El dataframe se separa en "X" e "y". Si hubiese una columna "y_gt" (folds PU), se descarta:
        if "y_gt" in dataframe.columns:
            X = dataframe.drop(columns = ["y", "y_gt"])
        else:
            X = dataframe.drop(columns = "y") # Evitamos que de error
        y = dataframe["y"]

    return X, y


# Función para juntar varios folds del mismo dataset en un solo dataset (para entrenamiento):
def juntar_folds_separados(ruta_folds, folds, guardado_npy = False):
    # Creamos dos listas para "X" e "y", que luego servirán para concatenar los folds:
    lista_X = []
    lista_y = []

    # Bucle principal:
    for fold_num in folds:
        # Se carga el fold:
        X, y = cargar_fold(ruta_folds, fold_num, guardado_npy)
        
        # Se añaden los valores de "X" e "y" a la lista
        lista_X.append(X)
        lista_y.append(y)

    # Si los folds estaban guardados como ".npy":
    if guardado_npy:
        # Se concatenan los arrays en un solo dataset:
        X = np.concatenate(lista_X, axis = 0)
        y = np.concatenate(lista_y, axis = 0)

    # Si los folds estaban guardados como ".csv":
    else:
        # Se concatenan los dataframes en un solo dataset:
        X = pd.concat(lista_X, ignore_index = True)
        y = pd.concat(lista_y, ignore_index = True)

    # El dataset no se guarda, se devuelve directamente:
    return X, y