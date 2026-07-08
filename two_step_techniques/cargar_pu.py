import numpy as np
import os

# Función para cargar un dataset convertido a PU en el preprocesado:
def cargar_dataset_pu(path_data, formato = "csv"):
    if formato == "npy":
        X = np.load(os.path.join(path_data, "X.npy"))
        y = np.load(os.path.join(path_data, "y.npy"))
        y_gt = np.load(os.path.join(path_data, "y_gt.npy"))
    
    if formato == "csv":
        # Nos saltamos la primera fila, que contiene la cabecera:
        data = np.loadtxt(path_data, delimiter = ",", skiprows = 1)

        # El archivo .csv se ha guardado con formato " X | y_gt | y_pu ".
        # Todas las columnas corresponden a "X", menos la penúltima ("y_gt") y la última ("y"):
        X = data[:, :-2]
        y_gt = data[:, -2]
        y = data[:, -1]
    
    return X, y, y_gt