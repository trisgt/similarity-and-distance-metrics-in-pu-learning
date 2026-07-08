import numpy as np
from codecarbon import EmissionsTracker

from config import ENGINEERED_DATASETS, SPLIT_DATASETS, RESULTADOS
from preprocesado.pr_bank_marketing import cargar_bank_marketing, CL_POSITIVAS, PORCENTAJE_POS
from two_step_techniques.cargar_pu import cargar_dataset_pu
from two_step_techniques.negativos_fiables import rocchio, knn, kmeans, kmedoids, crne
from two_step_techniques.aprendizaje import logistic_regression, random_forest, xgboost_lrn, mlp
from evaluacion.evaluacion import evaluacion_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Selección de métodos para Negativos Fiables y Aprendizaje:
MET_NEG_FIABLES = "Rocchio"
MET_APRENDIZAJE = "Random Forest"

# Parámetros para Negativos Fiables:
METRICA = "euclidean"
K = 10                  # En KNN, K-Means, K-Medoids y CRNE
PORCENTAJE_RN = 0.25    # En KNN, K-Means y K-Medoids
SEMILLA_RN = 1          # En K-Means, K-Medoids y CRNE

# Parámetros para Aprendizaje:
PENALTY = "l2"          # Con Regresión Logística
C = 10                  # Con Regresión Logística
N_ESTIMATORS = 100      # Con Random Forest
CRITERION = "gini"      # Con Random Forest
MAX_DEPTH = 100         # Con Random Forest
HIDDEN_LAYER_SIZES = (10, 10)   # Con Perceptrón Multicapa
ACTIVATION = "tanh"     # Con Perceptrón Multicapa
SEMILLA_L = 1

# Rutas del dataset de test y del dataset PU:
RUTA_DT_TEST = SPLIT_DATASETS / "bank+marketing" / "bank-additional" / "bank-additional-full-test.csv"
RUTA_DT_PU = ENGINEERED_DATASETS / "bank_marketing" / "dataset.csv"

# Ruta y nombre del .txt para el modelo "baseline" (PU sin Two-Step Methods):
RUTA_TXT_BASELINE = RESULTADOS / "bank_marketing_baseline.txt"
NOMBRE_BASELINE = "Bank Marketing Baseline"

# Ruta y nombre del .txt para el modelo (PU con Two-Step Methods):
RUTA_TXT = RESULTADOS / "bank_marketing.txt"
NOMBRE = "Bank Marketing"


# Función principal:
if __name__ == "__main__":
    # Carga del dataset PU (conj. de entrenamiento) y el de test:
    X, y, y_gt = cargar_dataset_pu(RUTA_DT_PU, "csv")
    X_test, y_test = cargar_bank_marketing(RUTA_DT_TEST)

    # PU sin Two-Step Methods ("Baseline"):

    # Se elige el modelo para el baseline:
    match MET_APRENDIZAJE:
        case "Logistic Regression":
            modelo_baseline = LogisticRegression(
                penalty = PENALTY,
                C = C,
                random_state = SEMILLA_L,
                max_iter = 1000
            )
        case "Random Forest":
            modelo_baseline = RandomForestClassifier(
                n_estimators = N_ESTIMATORS,
                criterion = CRITERION,
                max_depth = MAX_DEPTH,
                random_state = SEMILLA_L
            )
        case "XGBoost":
            modelo_baseline = XGBClassifier(
                random_state = SEMILLA_L
            )
        case "MLP":
            modelo_baseline = MLPClassifier(
                hidden_layer_sizes = HIDDEN_LAYER_SIZES,
                activation = ACTIVATION,
                max_iter = 1000,
                random_state = SEMILLA_L
            )
        case "CNN":
            raise ValueError("CNN aun no implementado")
        case _:
            raise ValueError(f"Modelo de aprendizaje no valido: \"{MET_APRENDIZAJE}\"")
    
    # Entrenamiento del modelo seleccionado:
    modelo_baseline.fit(X, y)

    # El modelo entrenado se utiliza para predecir las clases del conjunto de test:
    y_pred_baseline = modelo_baseline.predict(X_test)

    # Se evalúa el modelo:
    y_true_pu = np.isin(y_test, CL_POSITIVAS).astype(int)
    evaluacion_dataset(
        y_true_pu,
        y_pred_baseline,
        NOMBRE_BASELINE,
        CL_POSITIVAS,
        PORCENTAJE_POS,
        METRICA,
        MET_NEG_FIABLES,
        MET_APRENDIZAJE,
        True,
        RUTA_TXT_BASELINE
    )


    # PU con Two-Step Methods:

    # Codecarbon:
    #tracker = EmissionsTracker()
    #tracker.start()

    # Búsqueda de Negativos Fiables:
    match MET_NEG_FIABLES:
        case "Rocchio":
            RN = rocchio(X, y, METRICA)
        case "KNN":
            RN = knn(X, y, METRICA, K, PORCENTAJE_RN)
        case "KMeans":
            RN = kmeans(X, y, METRICA, K, PORCENTAJE_RN, SEMILLA_RN)
        case "KMedoids":
            RN = kmedoids(X, y, METRICA, K, PORCENTAJE_RN, SEMILLA_RN)
        case "CRNE":
            RN = crne(X, y, METRICA, K, SEMILLA_RN)
        case _:
            raise ValueError(f"Modelo de Negativos Fiables no valido: \"{MET_NEG_FIABLES}\"")
    
    # Aprendizaje con Positivos y Negativos Fiables:
    match MET_APRENDIZAJE:
        case "Logistic Regression":
            modelo = logistic_regression(X, y, RN, PENALTY, C, SEMILLA_L)
        case "Random Forest":
            modelo = random_forest(X, y, RN, N_ESTIMATORS, CRITERION, MAX_DEPTH, SEMILLA_L)
        case "XGBoost":
            modelo = xgboost_lrn(X, y, RN, SEMILLA_L)
        case "MLP":
            modelo = mlp(X, y, RN, HIDDEN_LAYER_SIZES, ACTIVATION, SEMILLA_L)
        case "CNN":
            raise ValueError("CNN aun no implementado")
        case _:
            raise ValueError(f"Modelo de aprendizaje no valido: \"{MET_APRENDIZAJE}\"")
    
    # El modelo entrenado se utiliza para predecir las clases del conjunto de test:
    y_pred = modelo.predict(X_test)

    # Se evalúa el modelo:
    evaluacion_dataset(
        y_true_pu,
        y_pred,
        NOMBRE,
        CL_POSITIVAS,
        PORCENTAJE_POS,
        METRICA,
        MET_NEG_FIABLES,
        MET_APRENDIZAJE,
        True,
        RUTA_TXT
    )

    # Codecarbon:
    #emissions = tracker.stop()
    #print(f"Emisiones: {emissions} kg CO2")