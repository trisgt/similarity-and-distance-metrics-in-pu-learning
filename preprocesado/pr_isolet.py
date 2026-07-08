import numpy as np

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from config import DATASETS, ENGINEERED_DATASETS

try:
    from preprocesado.preprocesado_dt import convertir_a_pu, guardar_dataset
except ModuleNotFoundError:
    from preprocesado_dt import convertir_a_pu, guardar_dataset

# Rutas del dataset original de entrenamiento y del dataset convertido a PU:
RUTA_DT_TRAIN = DATASETS / "isolet" / "isolet1+2+3+4.data"
RUTA_DT_PU = ENGINEERED_DATASETS / "isolet"

# Parámetros para el preprocesado:
CL_POSITIVAS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
PORCENTAJE_POS = 0.6
SEMILLA_PR = 1

# Función para cargar el dataset original (en este caso, de entrenamiento):
def cargar_isolet(path_data):
    data = np.loadtxt(path_data, delimiter = ",")

    # "X" son todas las columnas excepto la última, que es "y":
    X = data[:, :-1]
    y = data[:, -1]

    return X, y

# Función principal:
if __name__ == "__main__":
    X_train, y_train = cargar_isolet(RUTA_DT_TRAIN) # Carga del dataset

    # Preprocesado:
    X, y, y_gt = convertir_a_pu(X_train, y_train, CL_POSITIVAS, PORCENTAJE_POS, SEMILLA_PR)
    guardar_dataset(X, y, y_gt, RUTA_DT_PU, False, True)