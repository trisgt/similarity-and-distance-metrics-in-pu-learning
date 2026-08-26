import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from preprocesado.pu_engineering import convertir_a_pu, guardar_dataset_pu


# Función auxiliar para obtener los datos correspondientes a por unos índices.
# Es compatible tanto con datasets tabulares como de imágenes:
def seleccionar_indices(datos, indices):
    # Si los datos tienen en atributo "iloc" (propiedad de pandas), se utiliza:
    if hasattr(datos, "iloc"):
        return datos.iloc[indices]

    # En caso contrario (numpy), se devuelven sin utilizar "iloc":
    return datos[indices]


# Función para separar un dataset en K-folds:
def separar_dataset_en_k(X, y, k, ruta_guardado, guardado_npy, semilla = None):
    # Se divide el set en K-folds estratificados. Con ello se mantiene la misma
    # proporción de muestras de cada clase en cada división, aproximadamente:
    kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = semilla)

    # Se crea el directorio donde se guardarán los folds:
    os.makedirs(ruta_guardado, exist_ok = True)

    # Bucle principal:
    fold_num = 1
    for idx in kfold.split(X, y):
        # Sólo nos interesan los índices correspondientes al fold (conjunto de test) de cada división:
        fold_idx = idx[1]

        # Se obtienen los datos de "X" e "y" correspondientes a los índices dados:
        X_fold = seleccionar_indices(X, fold_idx)
        y_fold = seleccionar_indices(y, fold_idx)

        # Guardamos el fold en ".npy" o en ".csv", dependiendo de la opción indicada:
        guardar_fold(X_fold, y_fold, ruta_guardado, fold_num, guardado_npy)

        fold_num += 1
    
    print(f"Se han creado {k} folds en: {ruta_guardado}")


# Función auxiliar para guardar cada uno de los folds, tanto a ".csv" como a ".npy":
def guardar_fold(X, y, ruta, fold_num, guardado_npy):
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
        dataframe["y"] = np.asarray(y)

        # Guardamos el archivo .csv con la anterior ruta:
        dataframe.to_csv(ruta_csv, index = False, sep = ";")


# Función para generar folds PU a partir de los folds originales:
def generar_folds_pu(ruta_original, ruta_pu, guardado_npy, k, clases_positivas, porcentaje_positivos, semilla = None):
    # Se crea el directorio donde se guardarán las etiquetas PU:
    ruta_pu = Path(ruta_pu)
    os.makedirs(ruta_pu, exist_ok = True)

    # Bucle principal:
    for fold_num in range(1, k + 1):
        # Se carga el fold:
        X, y = cargar_fold(ruta_original, fold_num, guardado_npy)

        # Se convierte el fold a PU. Para ello, sólo es necesario convertir las etiquetas ("y" a "y_pu"):
        y_pu = convertir_a_pu(y, clases_positivas, porcentaje_positivos, semilla, f"Fold {fold_num}")

        # Se guarda el fold en la ruta deseada:
        guardar_dataset_pu(X, y_pu, ruta_pu, guardado_npy, f"fold_{fold_num}")
    
    print(f"Se han creado {k} folds PU en: {ruta_pu}")


# Función para juntar varios folds del mismo dataset en un solo dataset (para entrenamiento):
def juntar_folds_separados(ruta_folds, folds, guardado_npy):
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


# Función auxiliar para cargar un fold de un dataset, tanto ".csv" como ".npy":
def cargar_fold(ruta_folds, fold_num, guardado_npy, es_train = False, es_test = False):
    # Establecemos el nombre del fold. Si este es específicamente un fold de
    # entrenamiento o de test, modificamos el nombre adecuadamente:
    if es_train:
        nombre_fold = f"fold_{fold_num}_train"
    elif es_test:
        nombre_fold = f"fold_{fold_num}_test"
    else:
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
        # Se obtiene la ruta del fold como un Path:
        ruta_fold = Path(ruta_folds) / f"{nombre_fold}.csv"

        # Se crea un dataframe a partir del fold:
        dataframe = pd.read_csv(ruta_fold, sep = ";")

        # Separamos el dataframe en "X" e "y":
        X = dataframe.drop(columns = "y")
        y = dataframe["y"]

    return X, y