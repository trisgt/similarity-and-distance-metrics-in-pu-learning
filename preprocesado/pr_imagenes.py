import numpy as np
from pathlib import Path
from PIL import Image
import idx2numpy

# Función para cargar un conjunto de imágenes en formato ".png", ".jpg"...,
# donde cada clase tiene su carpeta de imágenes spropia:
def cargar_imgs_carpetas(path_data, clases, escala_grises = False):
    lista_X = []
    lista_y = []

    # Se recorren todas las clases:
    for nombre_clase, etiqueta in clases.items():
        ruta_clase = Path(path_data) / nombre_clase

        # Se recorren todas las imágenes de la clase:
        for ruta_imagen in sorted(ruta_clase.iterdir()):
            # Se comprueba que la ruta corresponda a un archivo (imágen). Si no, se descarta:
            if not ruta_imagen.is_file():
                continue

            # Se abre la imágen:
            with Image.open(ruta_imagen) as imagen:
                # La imágen se convierte o a escala de grises o a RGB:
                if escala_grises:
                    imagen = imagen.convert("L")
                else:
                    imagen = imagen.convert("RGB")

                # La imágen se convierte a un array de numpy. Después de este paso,
                # la imágen original se puede cerrar (fin del "with"):
                imagen = np.asarray(imagen, dtype = np.float32)

            # Si la imágen tiene solo dos dimensiones (altura, anchura), se añade una tercera (canales):
            if imagen.ndim == 2:
                imagen = np.expand_dims(imagen, axis = -1)

            # Se aplica normalización de los píxeles:
            imagen /= 255.0

            lista_X.append(imagen)
            lista_y.append(etiqueta)

    # Todas las imágenes se juntan en un único array:
    X = np.stack(lista_X)
    y = np.asarray(lista_y, dtype = int)

    return X, y


# Función para cargar imágenes en formato IDX binario, utilizando la librería "idx2numpy":
def cargar_imgs_idx(path_data):
    # Se convierte el Path en un string:
    path_data = str(path_data)

    # Se lee el archivo IDX y se convierte directamente a un "numpy.ndarray":
    X = idx2numpy.convert_from_file(path_data)

    # Los píxeles se convierten a float32 y se normalizan:
    X = X.astype(np.float32)
    X /= 255.0

    # Se añade una dimensión de canal:
    X = np.expand_dims(X, axis = -1)

    return X


# Función para cargar etiquetas en formato IDX binario, utilizando la librería "idx2numpy":
def cargar_etiqs_idx(path_data):
    # Se convierte el Path en un string:
    path_data = str(path_data)

    # Se lee el archivo IDX y se convierte directamente a un "numpy.ndarray":
    y = idx2numpy.convert_from_file(path_data)

    return y