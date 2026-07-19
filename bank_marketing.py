import numpy as np
import pandas as pd
from pathlib import Path
from codecarbon import EmissionsTracker

from config.paths_config import *
from preprocesado.kfold import juntar_folds_separados
from preprocesado.pr_bank_marketing import cargar_bank_marketing, NUM_FOLDS, CL_POSITIVAS, PORCENTAJE_POS
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
BALANCEO_CLASES = True

# Rutas para el dataset de entrenamiento (PU) y de test (no PU), en folds:
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


# Función auxiliar para calcular los pesos de las clases a partir de el conjunto de entrenamiento:
def calcular_pesos(y_train):
    # Calculamos el número de ejemplos de cada clase:
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)

    # Calculamos el peso para cada clase:
    peso_pos = len(y_train) / (num_pos * 2)
    peso_neg = len(y_train) / (num_neg * 2)
    pesos = np.where(y_train == 1, peso_pos, peso_neg)

    return pesos


# Función auxiliar para escribir en el archivo de resultados:
def escribir_resultados(metricas, ruta_guardado, nombre):
    # Las métricas se convierten a un DataFrame de pandas:
    metricas_dataframe = pd.DataFrame(metricas)

    # Se abre el archivo donde se guardarán los resultados:
    f = open(ruta_guardado, "a")

    # Se calculan la media y la desviación típica de las métricas:
    metricas_media = metricas_dataframe.mean()
    metricas_dt = metricas_dataframe.std()

    # Se escriben en el archivo los resultados:
    f.write(f"{nombre}:")
    f.write("\nMEDIA:\n")
    f.write(metricas_media.to_string())
    f.write("\nDESVIACION TIPICA:\n")
    f.write(metricas_dt.to_string())
    f.write("\n\n")

    # El archivo se cierra:
    f.close()


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



    # Entrenamiento y Evaluación no PU:
    print("\nEntrenamiento y Evaluacion no PU:\n")

    # Iniciamos CodeCarbon:
    tracker_benchmark = EmissionsTracker(
        project_name = "Benchmark",
        output_dir = RESULTADOS,
        output_file = "codecarbon.csv"
    )
    tracker_benchmark.start()

    # Bucle principal:
    print("\nEntrenamiento y Evaluacion no PU:")
    for fold_num in range(1, NUM_FOLDS + 1):
        print(f"\nFold {fold_num}...")

        # Carga del dataset de test (solo uno de los folds, no PU):
        X_test, y_test = cargar_bank_marketing(RUTA_FOLDS / f"fold_{fold_num}.csv")

        # Se juntan el resto de folds para formar el dataset de entrenamiento, no PU:
        indices_folds_entrenamiento = [i for i in range(1, NUM_FOLDS + 1) if i != fold_num]
        X_train, y_train = juntar_folds_separados(RUTA_FOLDS, indices_folds_entrenamiento)    

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

        print(f"Fold {fold_num} completado.")
    
    # Detenemos CodeCarbon tras la evaluación:
    emisiones_benchmark = tracker_benchmark.stop()

    print("Evaluacion sin PU completada")
    print(f"Emisiones: {emisiones_benchmark} kg CO2.\n")



    # Entrenamiento y Evaluación PU sin Two-Step Methods:
    print("\nEntrenamiento y Evaluacion PU sin Two-Step Methods:\n")

    # Iniciamos CodeCarbon:
    tracker_baseline = EmissionsTracker(
        project_name = "Baseline",
        output_dir = RESULTADOS,
        output_file = "codecarbon.csv"
    )
    tracker_baseline.start()

    # Bucle principal:
    for fold_num in range(1, NUM_FOLDS + 1):
        print(f"\nFold {fold_num}...")

        # Carga del dataset de test (solo uno de los folds, no PU):
        X_test, y_test = cargar_bank_marketing(RUTA_FOLDS / f"fold_{fold_num}.csv")

        # Se juntan el resto de folds para formar el dataset de entrenamiento, en este caso PU:
        indices_folds_entrenamiento = [i for i in range(1, NUM_FOLDS + 1) if i != fold_num]
        X_train_pu, y_train_pu = juntar_folds_separados(RUTA_FOLDS_PU, indices_folds_entrenamiento)

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

        # Si la opción está activada, balanceamos los clases. Si no, entrenamos sin balancear:
        if BALANCEO_CLASES:
            pesos = calcular_pesos(y_train_pu) # Calculamos los pesos de cada clase
            modelo_baseline.fit(X_train_pu, y_train_pu, sample_weight = pesos)
        else:
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

        print(f"\nFold {fold_num} completado.")

    # Detenemos CodeCarbon tras la evaluación:
    emisiones_baseline = tracker_baseline.stop()

    print("Evaluacion PU sin Two-Step Methods completada")
    print(f"Emisiones: {emisiones_baseline} kg CO2.\n")



    # Entrenamiento y Evaluación PU con Two-Step Methods:
    print("\nEntrenamiento y Evaluacion PU con Two-Step Methods:\n")

    # Iniciamos CodeCarbon:
    tracker = EmissionsTracker(
        project_name = "Two-Step",
        output_dir = RESULTADOS,
        output_file = "codecarbon.csv"
    )
    tracker.start()

    # Bucle principal:
    for fold_num in range(1, NUM_FOLDS + 1):
        print(f"\nFold {fold_num}:")

        # Carga del dataset de test (solo uno de los folds, no PU):
        X_test, y_test = cargar_bank_marketing(RUTA_FOLDS / f"fold_{fold_num}.csv")

        # Se juntan el resto de folds para formar el dataset de entrenamiento, PU:
        indices_folds_entrenamiento = [i for i in range(1, NUM_FOLDS + 1) if i != fold_num]
        X_train_pu, y_train_pu = juntar_folds_separados(RUTA_FOLDS_PU, indices_folds_entrenamiento)

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
                modelo = logistic_regression(X_train_pu, y_train_pu, RN, PENALTY, C, SEMILLA_L, BALANCEO_CLASES)
            case "Random Forest":
                modelo = random_forest(X_train_pu, y_train_pu, RN, N_ESTIMATORS, CRITERION, MAX_DEPTH, SEMILLA_L, BALANCEO_CLASES)
            case "XGBoost":
                modelo = xgboost_lrn(X_train_pu, y_train_pu, RN, SEMILLA_L, BALANCEO_CLASES)
            case "MLP":
                modelo = mlp(X_train_pu, y_train_pu, RN, HIDDEN_LAYER_SIZES, ACTIVATION, SEMILLA_L, BALANCEO_CLASES)
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

        print(f"\nFold {fold_num} completado.")
    
    # Detenemos CodeCarbon tras la evaluación:
    emisiones = tracker.stop()

    print("Evaluacion PU con Two-Step Methods completada")
    print(f"Emisiones: {emisiones} kg CO2.\n")



    # Se escriben las estadísticas finales en el archivo de guardado:
    escribir_resultados(metricas_benchmark, RUTA_TXT, "Resultados no PU")
    escribir_resultados(metricas_baseline, RUTA_TXT, "Resultados PU sin Two-Step Methods")
    escribir_resultados(metricas_two_step, RUTA_TXT, "Resultados PU con Two-Step Methods")

    print("Evaluacion finalizada correctamente.")
    print(f"Informe guardado en: {RUTA_TXT}")