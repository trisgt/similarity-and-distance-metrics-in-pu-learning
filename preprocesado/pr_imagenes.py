import numpy as np
from pathlib import Path
from PIL import Image

# Función para cargar un conjunto de imágenes.
# Funciona cuando cada clase tiene su carpeta propia:
def cargar_imagenes(path_data, clases, escala_grises = False):
    lista_X = []
    lista_y = []

    # Se recorren todas las clases:
    for nombre_clase, etiqueta in clases.items():
        ruta_clase = Path(path_data) / nombre_clase

        # Se recorren las imágenes de la clase:
        for ruta_imagen in sorted(ruta_clase.iterdir()):
            # Si la ruta no corresponde a un archivo (imágen), se descarta:
            if not ruta_imagen.is_file():
                continue

            with Image.open(ruta_imagen) as imagen:
                # La imágen se convierte o a escala de grises o a RGB:
                if escala_grises:
                    imagen = imagen.convert("L")
                else:
                    imagen = imagen.convert("RGB")

                # La imágen se convierte a array de numpy. Después, la imágen original se puede cerrar:
                imagen = np.asarray(imagen, dtype = np.float32)

            # Si solo tiene dos dimensiones (altura, anchura), se añade una tercera (canales):
            if imagen.ndim == 2:
                imagen = np.expand_dims(imagen, axis = -1)

            imagen = imagen / 255.0 # Normalización

            lista_X.append(imagen)
            lista_y.append(etiqueta)

    # Todas las imágenes se convierten en un único array:
    X = np.stack(lista_X)
    y = np.asarray(lista_y, dtype = int)

    return X, y