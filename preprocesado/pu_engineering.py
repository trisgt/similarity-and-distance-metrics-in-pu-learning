import numpy as np
import os


# Función principal para transformar un set de datos a uno de datos PU ("engineered"),
# donde "X" es la matriz con características (y muestras), e "y" son las etiquetas (ground truth):
def convertir_a_pu(X, y, clases_positivas, porcentaje_positivos, semilla = None, nombre_fold = None):
    # "X" e "y" se pasan como dataframes de pandas. Solo "y" se convierte a un array de Numpy:
    y = np.asarray(y)

    # Convertimos las clases positivas a un array de numpy. Si la clase fuese solo una (un escalar),
    # nos aseguramos de que sea iterable convirtiéndola en una lista antes de ello:
    if np.isscalar(clases_positivas):
        clases_positivas = [clases_positivas]
    clases_positivas = np.array(clases_positivas)

    # Generamos randomización, que puede ser controlada con una semilla específica.
    # Si la semilla no existe ("None"), cada iteración tendrá una randomización distinta:
    randomizador = np.random.default_rng(semilla)

    # (El dataset no se mezcla ahora para evitar que los folds PU sean distintos)

    # Aislamos los positivos y calculamos su número:
    pos_idx = np.where(np.isin(y, clases_positivas))[0]
    num_pos = len(pos_idx)

    # Con este número, calculamos el número de positivos que serán visibles en el dataset PU,
    # y seleccionamos arbitrariamente estos positivos del dataset:
    num_pos_sel = int(num_pos * porcentaje_positivos)
    pos_sel_idx = randomizador.choice(pos_idx, num_pos_sel, replace = False)

    # Ahora, se crean nuevas etiquetas "y", donde sólo los positivos seleccionados se marcan como clase positiva:
    y_pu = np.zeros(len(y), dtype = int)
    y_pu[pos_sel_idx] = 1

    # El proceso se ha realizado correctamente, y se le enseña información al usuario:
    print("\nDataset convertido a PU correctamente.")
    info_pu_engineering(y_pu, num_pos, num_pos_sel, nombre_fold)

    # Devolvemos "X" e "y" del dataset PU, además del "y" original:
    return X, y_pu, y


# Función auxiliar simplificada para resumir el estado de cada fold tras PU engineering:
def info_pu_engineering(y, num_pos_reales, num_pos_pu, nombre_fold = None):
    print(f"\n{nombre_fold}:")

    # Número de muestras del fold:
    num_muestras = len(y)
    print(f"\tMuestras: {num_muestras}")

    # Número de positivos reales:
    porcentaje_pos_reales = (num_pos_reales / num_muestras) * 100
    print(f"\tPositivos reales: {num_pos_reales} ({porcentaje_pos_reales:.2f} %)")

    # Número de positivos tras PU engineering:
    porcentaje_pos_pu = (num_pos_pu / num_muestras) * 100
    print(f"\tPositivos tras PU engineering : {num_pos_pu} ({porcentaje_pos_pu:.2f} %)")

    # Porcentaje de positivos etiquetados tras PU engineering:
    porcentaje_etiquetado = (num_pos_pu / num_pos_reales) * 100
    print(f"\tRatio etiquetado: {porcentaje_etiquetado:.2f} %\n")


# Función para guardar el dataset generado por "convertir_a_pu" en una carpeta.
# Los archivos se pueden guardar como ".npy" o como ".csv":
def guardar_dataset_pu(X, y, y_gt, carpeta_salida, guardado_npy, nombre):
    # Primero, creamos la carpeta si aún no existiese:
    os.makedirs(carpeta_salida, exist_ok = True)

    # Si la opción está activada, guardamos los archivos como ".npy".
    # En datasets de imágenes, se deben guardar como ".npy" y no como ".csv":
    if guardado_npy:
        np.save(os.path.join(carpeta_salida, f"{nombre}_X.npy"), X)
        np.save(os.path.join(carpeta_salida, f"{nombre}_y.npy"), y)
        np.save(os.path.join(carpeta_salida, f"{nombre}_y_gt.npy"), y_gt)

        print(f"Dataset .npy guardado en: {carpeta_salida}\n")
    
    # Si no, guardamos los archivos como ".csv":
    else:
        # Creamos un dataframe de pandas:
        dataframe = X.copy()

        # Se unen las nuevas columnas "y" e "y_gt" al dataframe, sin tener que concatenar manualmente:
        dataframe["y_gt"] = np.asarray(y_gt)
        dataframe["y"] = np.asarray(y)

        # Se define la ruta para el archivo:
        ruta_csv = os.path.join(carpeta_salida, f"{nombre}.csv")

        # Por último, se guarda el archivo CSV con la función "to_csv" de pandas:
        dataframe.to_csv(ruta_csv, index = False, sep = ";")

        print(f"Dataset .csv guardado en: {ruta_csv}\n")