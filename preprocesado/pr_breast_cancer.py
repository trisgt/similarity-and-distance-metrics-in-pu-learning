import os
import shutil
import pandas as pd
from config.paths_config import DATASETS, SPLIT_DATASETS, GENUINE_PU_DATASETS
from config.preproc_config import PORCENTAJE_POS

# Ruta de la carpeta con todos los datasets originales:
RUTA_CARPETA_DT = DATASETS / "Breast Cancer Wisconsin"

# Ruta de los folds del dataset binario (no PU) reorganizado:
RUTA_FOLDS = SPLIT_DATASETS / "breast_cancer"

# Ruta de los folds del dataset PU genuíno, con porcentaje de positivos escogido:
RUTA_FOLDS_GEN_PU = GENUINE_PU_DATASETS / "breast_cancer"

# Parámetros no modificables, específicos de Breast Cancer:
NUM_FOLDS = 5 # Número de folds del dataset
GUARDADO_NPY = False # Datasets guardados como ".csv"
CL_POSITIVAS = 1 # Clases consideradas como positivas


# Función auxiliar para convertir el porcentaje de positivos visibles usado en los experimentos
# al de positivos no etiquetados (n.e.) multiplicado por 100, el utilizado por Breast Cancer:
def obtener_porcentaje_pos_ne(porcentaje_pos):
    # Si el porcentaje de positivos es "None", nos referimos al dataset binario:
    if porcentaje_pos is None:
        return None

    # Porcentaje de positivos no etiquetados:
    porcentaje_pos_ne = 100 - (porcentaje_pos * 100)
    return int(porcentaje_pos_ne)


# Función auxiliar para obtener la carpeta del dataset utilizando el procentaje de positivos n.e. dado:
def obtener_carpeta_porc_pos_ne(porcentaje_pos_ne):
    # Si el porcentaje de positivos es "None", nos referimos al dataset binario:
    if porcentaje_pos_ne is None:
        return RUTA_CARPETA_DT / "Binary"

    # Cada carpeta utiliza la siguiente ruta:
    ruta_carpeta_porc_pos_ne = RUTA_CARPETA_DT / f"{porcentaje_pos_ne}%"
    return ruta_carpeta_porc_pos_ne


# Función auxiliar para construír el nombre de un fold, tanto de entrenamiento como de test:
def obtener_nombre_fold(porcentaje_pos_ne, fold_num, es_test):
    # Si el porcentaje de positivos es "None", nos referimos al dataset binario:
    if porcentaje_pos_ne is None:
        string_start = f"breast_cancer"
        string_end = f" set fold {fold_num}"
    else:
        string_start = f"breast_cancer_{porcentaje_pos_ne}_unlabelled"
        string_end = f" set fold {fold_num} random seed 42"

    # Obtenemos el nombre, dependiendo de si queremos un fold de entrenamiento o de test:
    if es_test:
        return f"{string_start}_test{string_end}.csv"
    else:
        return f"{string_start}_training{string_end}.csv"


# Función para reorganizar los folds originales y adaptarlos a la estructura del proyecto:
def reorganizar_folds(porcentaje_pos):
    # Convertimos el porcentaje de positivos al porcentaje de positivos no etiquetados:
    porcentaje_pos_ne = obtener_porcentaje_pos_ne(porcentaje_pos)

    # Obtenemos la ruta correspondiente al porcentaje actual:
    ruta_carpeta_porc_pos_ne = obtener_carpeta_porc_pos_ne(porcentaje_pos_ne)

    # Obtenemos, y creamos si fuese necesario, la ruta de destino:
    if porcentaje_pos is None:
        ruta_destino = RUTA_FOLDS
    else:
        ruta_destino = RUTA_FOLDS_GEN_PU
    os.makedirs(ruta_destino, exist_ok=True)

    # Recorremos cada fold:
    for fold_num in range(NUM_FOLDS):
        # Obtenemos los nombres de los folds para entrenamiento y test:
        nombre_fold_train = obtener_nombre_fold(porcentaje_pos_ne, fold_num, False)
        nombre_fold_test = obtener_nombre_fold(porcentaje_pos_ne, fold_num, True)

        # Obtenemos las rutas completas para estos folds:
        ruta_origen_train = ruta_carpeta_porc_pos_ne / nombre_fold_train
        ruta_origen_test = ruta_carpeta_porc_pos_ne / nombre_fold_test

        # Obtenemos las rutas de destino, utilizando los nombres convencionales de los experimentos.
        # Dependiendo de si el porcentaje es "None" (Binario) o no, se guarda en una carpeta u otra:
        if porcentaje_pos is None:
            ruta_destino_train = RUTA_FOLDS / f"fold_{fold_num + 1}_train.csv"
            ruta_destino_test = RUTA_FOLDS / f"fold_{fold_num + 1}_test.csv"
        else:
            ruta_destino_train = RUTA_FOLDS_GEN_PU / f"fold_{fold_num + 1}_train.csv"
            ruta_destino_test = RUTA_FOLDS_GEN_PU / f"fold_{fold_num + 1}_test.csv"

        # Copiamos el contenido de los archivos. Para mantener compatibilidad con el resto de funciones,
        # se cambia el separador original (",") a ";" y se renombra la columna "diagnosis" a "y":
        fold_train = pd.read_csv(ruta_origen_train, sep = ",")
        fold_train = fold_train.rename(columns = {"diagnosis": "y"})
        fold_train.to_csv(ruta_destino_train, sep = ";", index = False)

        # Igual con el contenido del archivo de test:
        fold_test = pd.read_csv(ruta_origen_test, sep = ",")
        fold_test = fold_test.rename(columns = {"diagnosis": "y"})
        fold_test.to_csv(ruta_destino_test, sep = ";", index = False)

    if porcentaje_pos is None:
        print(f"Folds binarios reorganizados correctamente en: {RUTA_FOLDS}")
    else:
        print(f"Folds PU reorganizados correctamente en: {RUTA_FOLDS_GEN_PU}")


# Función principal:
if __name__ == "__main__":
    # Reorganizamos los folds binarios:
    reorganizar_folds(None)

    # Reorganizamos los folds PU con el porcentaje de positivos n.e. seleccionado:
    reorganizar_folds(PORCENTAJE_POS)    