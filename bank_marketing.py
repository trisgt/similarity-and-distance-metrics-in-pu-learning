import numpy as np
import pandas as pd
from pathlib import Path
from codecarbon import EmissionsTracker

from config import ENGINEERED_DATASETS, SPLIT_DATASETS, RESULTADOS
from preprocesado.kfold import juntar_folds_separados
from preprocesado.pr_bank_marketing import cargar_bank_marketing, NUM_FOLDS, CL_POSITIVAS, PORCENTAJE_POS
from two_step_techniques.cargar_pu import cargar_dataset_pu
from two_step_techniques.negativos_fiables import rocchio, knn, kmeans, kmedoids, crne
from two_step_techniques.aprendizaje import logistic_regression, random_forest, xgboost_lrn, mlp
from evaluacion.evaluacion import informacion_evaluacion, evaluacion_dataset

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
N_ESTIMATORS = 50       # Con Random Forest
CRITERION = "gini"      # Con Random Forest
MAX_DEPTH = 50          # Con Random Forest
HIDDEN_LAYER_SIZES = (10, 10)   # Con Perceptrón Multicapa
ACTIVATION = "tanh"     # Con Perceptrón Multicapa
SEMILLA_L = 1

# Rutas para eel dataset de entrenamiento (PU) y de test (no PU), en folds:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "bank_marketing"
RUTA_FOLDS = SPLIT_DATASETS / "bank_marketing"

# Ruta y nombre del .txt para los resultados:
RUTA_TXT = RESULTADOS / "bank_marketing.txt"
NOMBRE = "Bank Marketing"

# Función auxiliar para la elección de un modelo de aprendizaje (sin Two-Step):
def elegir_metodo_aprendizaje(
        metodo,
        penalty = "l2",
        c = 10,
        n_estimators = 10,
        criterion = "gini",
        max_depth = 100,
        hidden_layer_sizes = (10, 10),
        activation = "tanh",
        semilla = None):
    
    # Se elige el método
    match metodo:
        case "Logistic Regression":
            modelo = LogisticRegression(penalty = penalty, C = c, random_state = semilla, max_iter = 1000)
        case "Random Forest":
            modelo = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, random_state = semilla)
        case "XGBoost":
            modelo = XGBClassifier(random_state = semilla)
        case "MLP":
            modelo = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, activation = activation, max_iter = 1000, random_state = semilla)
        case "CNN":
            raise ValueError("CNN aun no implementado")
        case _:
            raise ValueError(f"Modelo de aprendizaje no valido: \"{metodo}\"")
    
    return modelo


# Función principal:
if __name__ == "__main__":
    # Se crean arrays para más tarde guardar los resultados:
    metricas_benchmark = []
    metricas_baseline = []
    metricas_two_step = []

    informacion_evaluacion(
        NOMBRE,
        CL_POSITIVAS,
        PORCENTAJE_POS,
        METRICA,
        MET_NEG_FIABLES,
        MET_APRENDIZAJE,
        True,
        RUTA_TXT
    )

    # Bucle principal:
    for fold_num in range(1, NUM_FOLDS + 1):
        print(f"\n\nFold {fold_num}:\n")

        # Carga del dataset de test (solo uno de los folds, no PU):
        X_test, y_test = cargar_bank_marketing(RUTA_FOLDS / f"fold_{fold_num}.csv")

        # Se juntan el resto de folds para formar el dataset de entrenamiento.
        # Se forman dos conjuntos: uno no PU (para el Benchmark) y otro PU:
        indices_folds_entrenamiento = [i for i in range(1, NUM_FOLDS + 1) if i != fold_num]
        X_train, y_train = juntar_folds_separados(RUTA_FOLDS, indices_folds_entrenamiento)
        X_train_pu, y_train_pu = juntar_folds_separados(RUTA_FOLDS_PU, indices_folds_entrenamiento)


        # Sin PU ("Benchmark"):

        # Se elige el modelo para el benchmark:
        modelo_benchmark = elegir_metodo_aprendizaje(
            MET_APRENDIZAJE,
            PENALTY,
            C,
            N_ESTIMATORS,
            CRITERION,
            MAX_DEPTH,
            HIDDEN_LAYER_SIZES,
            ACTIVATION,
            SEMILLA_L
        )
        
        # Se entrena el modelo:
        modelo_benchmark.fit(X_train, y_train)

        # El modelo entrenado se utiliza para predecir las clases del conjunto de test:
        y_pred_benchmark = modelo_benchmark.predict(X_test)

        # Obtenemos también las probabilidades que el modelo asigna a la clase positiva:
        y_score_benchmark = modelo_benchmark.predict_proba(X_test)[:, 1]

        # Se evalúa el modelo:
        resultados_benchmark = evaluacion_dataset(
            y_test,
            y_pred_benchmark,
            y_score_benchmark
        )

        # Se añaden los resultados de las métricas a la lista:
        metricas_benchmark.append(resultados_benchmark)

        print("Evaluación sin PU completada")


        # PU sin Two-Step Methods ("Baseline"):

        # Se elige el modelo para el baseline:
        modelo_baseline = elegir_metodo_aprendizaje(
            MET_APRENDIZAJE,
            PENALTY,
            C,
            N_ESTIMATORS,
            CRITERION,
            MAX_DEPTH,
            HIDDEN_LAYER_SIZES,
            ACTIVATION,
            SEMILLA_L
        )
    
        # Entrenamiento del modelo seleccionado:
        modelo_baseline.fit(X_train_pu, y_train_pu)

        # El modelo entrenado se utiliza para predecir las clases del conjunto de test:
        y_pred_baseline = modelo_baseline.predict(X_test)

        # Obtenemos también las probabilidades que el modelo asigna a la clase positiva:
        y_score_baseline = modelo_baseline.predict_proba(X_test)[:, 1]

        # Se evalúa el modelo:
        y_true_pu = np.isin(y_test, CL_POSITIVAS).astype(int)
        resultados_baseline = evaluacion_dataset(
            y_true_pu,
            y_pred_baseline,
            y_score_baseline
        )

        # Se añaden los resultados de las métricas a la lista:
        metricas_baseline.append(resultados_baseline)

        print("Evaluación PU completada")


        # PU con Two-Step Methods:

        # Codecarbon:
        #tracker = EmissionsTracker()
        #tracker.start()

        # Búsqueda de Negativos Fiables:
        match MET_NEG_FIABLES:
            case "Rocchio":
                RN = rocchio(X_train_pu, y_train_pu, METRICA)
            case "KNN":
                RN = knn(X_train_pu, y_train_pu, METRICA, K, PORCENTAJE_RN)
            case "KMeans":
                RN = kmeans(X_train_pu, y_train_pu, METRICA, K, PORCENTAJE_RN, SEMILLA_RN)
            case "KMedoids":
                RN = kmedoids(X_train_pu, y_train_pu, METRICA, K, PORCENTAJE_RN, SEMILLA_RN)
            case "CRNE":
                RN = crne(X_train_pu, y_train_pu, METRICA, K, SEMILLA_RN)
            case _:
                raise ValueError(f"Modelo de Negativos Fiables no valido: \"{MET_NEG_FIABLES}\"")
        
        # Aprendizaje con Positivos y Negativos Fiables:
        match MET_APRENDIZAJE:
            case "Logistic Regression":
                modelo = logistic_regression(X_train_pu, y_train_pu, RN, PENALTY, C, SEMILLA_L)
            case "Random Forest":
                modelo = random_forest(X_train_pu, y_train_pu, RN, N_ESTIMATORS, CRITERION, MAX_DEPTH, SEMILLA_L)
            case "XGBoost":
                modelo = xgboost_lrn(X_train_pu, y_train_pu, RN, SEMILLA_L)
            case "MLP":
                modelo = mlp(X_train_pu, y_train_pu, RN, HIDDEN_LAYER_SIZES, ACTIVATION, SEMILLA_L)
            case "CNN":
                raise ValueError("CNN aun no implementado")
            case _:
                raise ValueError(f"Modelo de aprendizaje no valido: \"{MET_APRENDIZAJE}\"")
        
        # El modelo entrenado se utiliza para predecir las clases del conjunto de test:
        y_pred = modelo.predict(X_test)

        # Obtenemos también las probabilidades que el modelo asigna a la clase positiva:
        y_score = modelo.predict_proba(X_test)[:, 1]

        # Se evalúa el modelo:
        y_true_pu = np.isin(y_test, CL_POSITIVAS).astype(int)
        resultados_two_step = evaluacion_dataset(
            y_true_pu,
            y_pred,
            y_score
        )

        # Se añaden los resultados de las métricas a la lista:
        metricas_two_step.append(resultados_two_step)

        # Codecarbon:
        #emissions = tracker.stop()
        #print(f"Emisiones: {emissions} kg CO2")

        print("Evaluación PU con Two-Step Methods completada")
    
    # Estadísticas finales:

    benchmark = pd.DataFrame(metricas_benchmark)
    baseline = pd.DataFrame(metricas_baseline)
    two_step = pd.DataFrame(metricas_two_step)

    # Se abre el archivo:
    f = open(RUTA_TXT, "a")

    # Resultados del benchmark:
    f.write("No PU:")
    f.write("\n\nMedia:\n")
    f.write(benchmark.mean().to_string())
    f.write("\n\tDesviacion tipica:\n\t")
    f.write(benchmark.std().to_string().replace("\n", "\n\t"))

    # Resultados del baseline:
    f.write("\n\nPU sin Two-Step Methods:")
    f.write("\n\nMedia:\n")
    f.write(baseline.mean().to_string())
    f.write("\n\tDesviacion tipica:\n\t")
    f.write(baseline.std().to_string().replace("\n", "\n\t"))

    # Resultados del two step:
    f.write("\n\nPU con Two-Step Methods:")
    f.write("\n\nMedia:\n")
    f.write(two_step.mean().to_string())
    f.write("\n\tDesviacion tipica:\n\t")
    f.write(two_step.std().to_string().replace("\n", "\n\t"))

    f.write("\n\n\n")

    print("Evaluacion finalizada correctamente.")
    print(f"Informe guardado en: {RUTA_TXT}")