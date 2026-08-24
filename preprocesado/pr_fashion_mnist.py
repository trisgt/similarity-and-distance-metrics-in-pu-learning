import numpy as np
from config.paths_config import DATASETS, ENGINEERED_DATASETS, SPLIT_DATASETS
from preprocesado.pr_imagenes import cargar_imgs_idx, cargar_etiqs_idx
from preprocesado.kfold import separar_dataset_en_k, generar_folds_pu

# Rutas del dataset original de entrenamiento y de test:
RUTA_DT_TRAIN = DATASETS / "fashion_mnist" / "train-images-idx3-ubyte"
RUTA_DT_TRAIN_LABELS = DATASETS / "fashion_mnist" / "train-labels-idx1-ubyte"
RUTA_DT_TEST = DATASETS / "fashion_mnist" / "t10k-images-idx3-ubyte"
RUTA_DT_TEST_LABELS = DATASETS / "fashion_mnist" / "t10k-labels-idx1-ubyte"

# Ruta del dataset tras separación en K-folds:
RUTA_FOLDS = SPLIT_DATASETS / "fashion_mnist"

# Ruta de los folds tras convertirlos en PU:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "fashion_mnist"


# Parámetros para el split del dataset en k-folds:
NUM_FOLDS = 5
SEMILLA_SPLIT = 1

# Parámetros para el guardado de los datasets:
GUARDADO_NPY = True # Si es falso, los datasets se guardan como ".csv"

# Parámetros para el preprocesado:
CL_POSITIVAS = [0, 1, 2]
PORCENTAJE_POS = 0.6
SEMILLA_PR = 1

# Función para cargar el dataset original. Se concatenan los de entrenamiento y test para luego hacer k-folds:
def cargar_fashion_mnist(path_train, path_train_labels, path_test, path_test_labels):
    # Como los conjuntos y sus labels ("y") están separados, hay que cargar 4 conjuntos:
    X_train = cargar_imgs_idx(path_train)
    X_test = cargar_imgs_idx(path_test)
    y_train = cargar_etiqs_idx(path_train_labels)
    y_test = cargar_etiqs_idx(path_test_labels)

    # Tras leer el dataset de entrenamiento y test, ambos se concatenan:
    X = np.concatenate([X_train, X_test], axis = 0)
    y = np.concatenate([y_train, y_test], axis = 0)

    return X, y

# Función principal:
if __name__ == "__main__":
    # Carga y concatenación de los dataset originales:
    X, y = cargar_fashion_mnist(RUTA_DT_TRAIN, RUTA_DT_TRAIN_LABELS, RUTA_DT_TEST, RUTA_DT_TEST_LABELS)

    # Separación en K-folds:
    separar_dataset_en_k(
        X,
        y,
        NUM_FOLDS,
        RUTA_FOLDS,
        GUARDADO_NPY,
        SEMILLA_SPLIT
    )

    # Transformación de los folds a PU. La conversión se hace dentro de
    # la propia función "generar_folds_pu":
    generar_folds_pu(
        RUTA_FOLDS,
        RUTA_FOLDS_PU,
        GUARDADO_NPY,
        NUM_FOLDS,
        CL_POSITIVAS,
        PORCENTAJE_POS,
        SEMILLA_PR
    )

    print("Preprocesado completado")