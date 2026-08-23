import numpy as np
import pandas as pd
import os


# Función principal para transformar un set de datos a uno de datus PU utilizando PU engineering,
# donde "y" son las etiquetas originales (ground truth). No es necesario usar "X" ya que no sería modificada:
def convertir_a_pu(y, clases_positivas, porcentaje_positivos, semilla = None, nombre_fold = None):
    # "y" se ha pasado como un dataframe de pandas. Ahora, se convierte en un array de Numpy:
    y = np.asarray(y)

    # Las clases positivas se convierten a un array de numpy. Además, en caso de que la clase
    # fuese solo una (un escalar), la hacemos iterable convirtiéndola en una lista:
    clases_positivas = np.atleast_1d(clases_positivas)

    # Generamos randomización, que puede ser controlada con una semilla específica.
    # Si la semilla no existe ("None"), cada iteración tendrá una randomización distinta:
    randomizador = np.random.default_rng(semilla)

    # Obtenemos los índices de las clases positivas, y calculamos el número de positivos:
    pos_idx = np.where(np.isin(y, clases_positivas))[0]
    num_pos = len(pos_idx)

    # Con este número, calculamos el número de positivos visibles (etiquetados) en el dataset PU,
    # y seleccionamos arbitrariamente estos positivos de entre los originales:
    num_pos_etiq = int(num_pos * porcentaje_positivos)
    pos_sel_idx = randomizador.choice(pos_idx, num_pos_etiq, replace = False)

    # Ahora, se crean nuevas etiquetas "y_pu". Todas empiezan como muestras sin etiqueta (0),
    # y sólo los positivos seleccionados se marcan como clase positiva (con etiqueta, 1):
    y_pu = np.zeros(len(y), dtype = int)
    y_pu[pos_sel_idx] = 1

    # El proceso se ha realizado correctamente, y se le enseña información al usuario:
    print("\nDataset convertido a PU correctamente.")
    info_pu_engineering(y_pu, num_pos, num_pos_etiq, nombre_fold)

    # Devolvemos "y_pu" ("y" convertida utilizando PU engineering).
    # No se ha trabajado con "X", ya que esta no se modificaría durante el proceso:
    return y_pu


# Función auxiliar simplificada para resumir el estado de cada fold tras PU engineering:
def info_pu_engineering(y, num_pos_reales, num_pos_etiq, nombre_fold = None):
    print(f"\n{nombre_fold}:")

    # Número de muestras:
    num_muestras = len(y)
    print(f"\tMuestras: {num_muestras}")

    # Número de positivos reales (originales), además de su porcentaje:
    porcentaje_pos_reales = (num_pos_reales / num_muestras) * 100
    print(f"\tPositivos reales: {num_pos_reales} ({porcentaje_pos_reales:.2f} %)")

    # Número de positivos etiquetados tras PU engineering, además de su porcentaje:
    porcentaje_pos_etiq = (num_pos_etiq / num_muestras) * 100
    print(f"\tPositivos tras PU engineering : {num_pos_etiq} ({porcentaje_pos_etiq:.2f} %)")

    # Porcentaje de los positivos reales etiquetados tras PU engineering:
    porcentaje_reales_etiq = (num_pos_etiq / num_pos_reales) * 100
    print(f"\tRatio etiquetado: {porcentaje_reales_etiq:.2f} %\n")


# Función para guardar el dataset con las etiquetas generadas por "convertir_a_pu" ("y_pu")
# en una carpeta. Los archivos se pueden guardar como ".npy" o como ".csv":
def guardar_dataset_pu(X, y_pu, carpeta_salida, guardado_npy, nombre):
    # Primero, creamos la carpeta de salida si esta aún no existiese:
    os.makedirs(carpeta_salida, exist_ok = True)

    # Si la opción está activada, guardamos los archivos como ".npy".
    # En datasets de imágenes, se deben guardar como ".npy" y no como ".csv":
    if guardado_npy:
        np.save(os.path.join(carpeta_salida, f"{nombre}_X.npy"), X)
        np.save(os.path.join(carpeta_salida, f"{nombre}_y.npy"), y_pu)

        print(f"Dataset .npy guardado en: {carpeta_salida}\n")
    
    # Si no, guardamos los archivos como ".csv":
    else:
        # Creamos un dataframe de pandas:
        dataframe = X.copy()

        # Añadimos una nueva columna "y" al dataframe:
        dataframe["y"] = np.asarray(y_pu)

        # Se define la ruta para el archivo:
        ruta_csv = os.path.join(carpeta_salida, f"{nombre}.csv")

        # Por último, se guarda el archivo CSV con la función "to_csv" de pandas:
        dataframe.to_csv(ruta_csv, index = False, sep = ";")

        print(f"Dataset .csv guardado en: {ruta_csv}\n")