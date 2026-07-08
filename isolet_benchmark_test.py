import numpy as np

from config import DATASETS, RESULTADOS
from preprocesado.pr_isolet import cargar_isolet, CL_POSITIVAS
from evaluacion.evaluacion import evaluacion_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Selección de método para Aprendizaje:
MET_APRENDIZAJE = "XGBoost"

# Parámetros para Aprendizaje:
PENALTY = "l2"          # Con Regresión Logística
C = 10                  # Con Regresión Logística
N_ESTIMATORS = 100      # Con Random Forest
CRITERION = "gini"      # Con Random Forest
MAX_DEPTH = 100         # Con Random Forest
HIDDEN_LAYER_SIZES = (10, 10)   # Con Perceptrón Multicapa
ACTIVATION = "tanh"     # Con Perceptrón Multicapa
SEMILLA_L = 1

# Rutas de los datasets de entrenamiento y test:
RUTA_DT_TRAIN = DATASETS / "isolet" / "isolet1+2+3+4.data"
RUTA_DT_TEST = DATASETS / "isolet" / "isolet5.data"

# Ruta y nombre del .txt para el modelo:
RUTA_TXT = RESULTADOS / "isolet_benchmark_text.txt"
NOMBRE = "Benchmark Test ISOLET"

# Programa principal. Este servirá como prueba de que el proceso funciona correctamente:
if __name__ == "__main__":
    # Carga de los datasets originales de entrenamiento y test:
    X_train, y_train = cargar_isolet(RUTA_DT_TRAIN)
    X_test, y_test = cargar_isolet(RUTA_DT_TEST)

    # Se binarizan los conjuntos de train y test:
    y_train = np.isin(y_train, CL_POSITIVAS).astype(int)
    y_test = np.isin(y_test, CL_POSITIVAS).astype(int)

    # Se elige el modelo:
    match MET_APRENDIZAJE:
        case "Logistic Regression":
            modelo = LogisticRegression(
                penalty = PENALTY,
                C = C,
                random_state = SEMILLA_L,
                max_iter = 1000
            )
        case "Random Forest":
            modelo = RandomForestClassifier(
                n_estimators = N_ESTIMATORS,
                criterion = CRITERION,
                max_depth = MAX_DEPTH,
                random_state = SEMILLA_L
            )
        case "XGBoost":
            modelo = XGBClassifier(
                random_state = SEMILLA_L
            )
        case "MLP":
            modelo = MLPClassifier(
                hidden_layer_sizes = HIDDEN_LAYER_SIZES,
                activation = ACTIVATION,
                max_iter = 1000,
                random_state = SEMILLA_L
            )
        case "CNN":
            raise ValueError("CNN aun no implementado")
        case _:
            raise ValueError(f"Modelo de aprendizaje no valido: \"{MET_APRENDIZAJE}\"")

    # Se entrena el modelo:
    modelo.fit(X_train, y_train)

    # El modelo entrenado se utiliza para predecir las clases del conjunto de test:
    y_pred = modelo.predict(X_test)

    evaluacion_dataset(
        y_test,
        y_pred,
        NOMBRE,
        met_aprendizaje = MET_APRENDIZAJE,
        guardar = True,
        path_guardado = RUTA_TXT
    )