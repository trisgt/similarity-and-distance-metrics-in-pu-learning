import numpy as np
import os


# Función principal para transformar un set de datos a uno de datos PU ("engineered"),
# donde "X" es la matriz con características (y muestras), e "y" son las etiquetas (ground truth):
def convertir_a_pu(X, y, clases_positivas, porcentaje_positivos, semilla = None):
    # Primero, convertimos "X" e "y" a arrays de Numpy:
    X = np.array(X)
    y = np.array(y)

    # Convertimos las clases positivas a un array de numpy. Si la clase fuese solo una (un escalar),
    # nos aseguramos de que sea iterable convirtiéndola en una lista antes de ello:
    if np.isscalar(clases_positivas):
        clases_positivas = [clases_positivas]
    clases_positivas = np.array(clases_positivas)

    # Generamos randomización, que puede ser controlada con una semilla específica.
    # Si la semilla no existe ("None"), cada iteración tendrá una randomización distinta:
    randomizador = np.random.default_rng(semilla)

    # Se mezcla el dataset para evitar sesgo:
    indices = np.arange(len(y))
    randomizador.shuffle(indices)
    X = X[indices] # Se reordena "X"
    y = y[indices] # Se reordena "y"

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
    print("Dataset convertido a PU correctamente")
    informacion_dataset(y_pu, y, clases_positivas, num_pos, num_pos_sel)

    # Devolvemos "X" e "y" del dataset PU, además del "y" original:
    return X, y_pu, y


# Función auxiliar para resumir el estado del dataset tras PU engineering:
def informacion_dataset(y, y_gt, clases_positivas, num_positivos, num_positivos_tras_pu):
    print("\nInformacion del dataset:")

    # Clases originales, con número de muestras de cada clase:
    print("\nClases originales:")
    clases_gt, num_muestras_gt = np.unique(y_gt, return_counts = True)
    for c, n in zip(clases_gt, num_muestras_gt):
        print(f"Clase {c}: {n}")
    print(f"Numero de clases: {len(clases_gt)}")

    # Clases seleccionadas como positivas para PU:
    print(f"\nClases seleccionadas: {clases_positivas}")
    print(f"Numero de clases seleccionadas: {len(clases_positivas)}")

    # Etiquetas tras PU engineering:
    print("\nEtiquetas tras PU engineering:")
    clases, num_muestras = np.unique(y, return_counts = True)
    for c, n in zip(clases, num_muestras):
        print(f"Etiqueta {c}: {n}")

    # Número de posirivos en el dataset original y tras PU engineering:
    print(f"\nNumero de positivos de las clases seleccionadas: {num_positivos}")
    print(f"Numero de positivos tras PU engineering: {num_positivos_tras_pu}")
    print(f"Ratio etiquetado tras PU engineering: {num_positivos_tras_pu / num_positivos}\n")


# Función para guardar el dataset generado por "convertir_a_pu" en una carpeta.
# Se puede escoger tanto si se guardan los archivos ".npy" como ".csv":
def guardar_dataset(X, y, y_gt, carpeta_salida, npy = False, csv = True):
    # Primero, creamos la carpeta si aún no existiese:
    os.makedirs(carpeta_salida, exist_ok = True)

    # Guardamos los archivos ".npy":
    if (npy):
        np.save(os.path.join(carpeta_salida, "X.npy"), X)
        np.save(os.path.join(carpeta_salida, "y.npy"), y)
        np.save(os.path.join(carpeta_salida, "y_gt.npy"), y_gt)

        print(f"Dataset .npy guardado en: {carpeta_salida}")
    
    # Guardamos los archivos ".csv":
    if (csv):
        # Convertimos "X", "y" e "y_gt" a arrays de Numpy ("y" e "y_gt" se transforman en columnas):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        y_gt = np.array(y_gt).reshape(-1, 1)

        # Se concatenan las matrices horizontalmente:
        datos = np.hstack([X, y_gt, y])

        # Se crea una cabecera para el archivo:
        num_features = X.shape[1]

        columnas = []
        for i in range(num_features):
            columnas.append(f"f{i}") # Se usan nombres genéricos para las características

        columnas += ["y", "y_gt"]
        header = ",".join(columnas)

        # Se define la ruta para el archivo:
        ruta_csv = os.path.join(carpeta_salida, "dataset.csv")

        # Por último, se guarda el archivo CSV con la función "savetxt":
        np.savetxt(
            ruta_csv,
            datos,
            delimiter = ",",
            header = header,
            comments = "",
            fmt = "%.5f"
        )

        print(f"Dataset .csv guardado en: {ruta_csv}")