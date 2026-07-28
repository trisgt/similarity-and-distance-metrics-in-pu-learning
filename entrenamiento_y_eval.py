import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

from config.paths_config import RESULTADOS
from config.train_config import *

from preprocesado.kfold import cargar_fold, juntar_folds_separados
from two_step_techniques.negativos_fiables import rocchio, knn, kmeans, kmedoids, crne
from two_step_techniques.aprendizaje import logistic_regression, random_forest, xgboost_lrn, mlp
from evaluacion.evaluacion import informacion_evaluacion, evaluacion_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


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


# Función de Entrenamiento y Evaluación no PU:
def entr_y_eval_no_pu(ruta_folds, num_folds, cl_positivas, metricas, nombre):
    print("\nEntrenamiento y Evaluacion no PU:\n")

    # Iniciamos CodeCarbon:
    tracker = EmissionsTracker(
        project_name = f"{nombre}, no PU",
        output_dir = RESULTADOS,
        output_file = "codecarbon.csv"
    )
    tracker.start()

    # Bucle principal:
    for fold_num in range(1, num_folds + 1):
        print(f"\nFold {fold_num}...")

        # Carga del dataset de test (solo uno de los folds, no PU):
        X_test, y_test = cargar_fold(ruta_folds / f"fold_{fold_num}.csv")

        # Se juntan el resto de folds para formar el dataset de entrenamiento, no PU:
        indices_folds_entrenamiento = [i for i in range(1, num_folds + 1) if i != fold_num]
        X_train, y_train = juntar_folds_separados(ruta_folds, indices_folds_entrenamiento)    

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

        # Se convierten las etiquetas multiclase a binarias:
        y_train = np.isin(y_train, cl_positivas).astype(int)
        y_test = np.isin(y_test, cl_positivas).astype(int)

        # Se entrena el modelo:
        modelo_benchmark.fit(X_train, y_train)

        # El modelo entrenado se utiliza para predecir las clases del conjunto de test:
        y_pred_benchmark = modelo_benchmark.predict(X_test)

        # Obtenemos también las probabilidades que el modelo asigna a la clase positiva:
        y_score_benchmark = modelo_benchmark.predict_proba(X_test)[:, 1]

        # Se evalúa el modelo:
        resultados_no_pu = evaluacion_dataset(
            y_test,
            y_pred_benchmark,
            y_score_benchmark
        )

        # Se añaden los resultados de las métricas a la lista:
        metricas.append(resultados_no_pu)

        print(f"Fold {fold_num} completado.")
    
    # Detenemos CodeCarbon tras la evaluación:
    emisiones = tracker.stop()

    print("Evaluacion no PU completada")
    print(f"Emisiones: {emisiones} kg CO2.\n")

    # Devolvemos los resultados no PU:
    return metricas


# Función de Entrenamiento y Evaluación PU sin Two-Step methods:
def entr_y_eval_pu_sin_ts(ruta_folds, ruta_folds_pu, num_folds, cl_positivas, metricas, nombre):
    print("\nEntrenamiento y Evaluacion PU sin Two-Step methods:\n")

    # Iniciamos CodeCarbon:
    tracker = EmissionsTracker(
        project_name = f"{nombre}, PU sin Two-Step",
        output_dir = RESULTADOS,
        output_file = "codecarbon.csv"
    )
    tracker.start()

    # Bucle principal:
    for fold_num in range(1, num_folds + 1):
        print(f"\nFold {fold_num}...")

        # Carga del dataset de test (solo uno de los folds, no PU):
        X_test, y_test = cargar_fold(ruta_folds / f"fold_{fold_num}.csv")

        # Se juntan el resto de folds para formar el dataset de entrenamiento, en este caso PU:
        indices_folds_entrenamiento = [i for i in range(1, num_folds + 1) if i != fold_num]
        X_train_pu, y_train_pu = juntar_folds_separados(ruta_folds_pu, indices_folds_entrenamiento)

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
        y_true_pu = np.isin(y_test, cl_positivas).astype(int)
        resultados_pu_sin_ts = evaluacion_dataset(
            y_true_pu,
            y_pred_baseline,
            y_score_baseline
        )

        # Se añaden los resultados de las métricas a la lista:
        metricas.append(resultados_pu_sin_ts)

        print(f"\nFold {fold_num} completado.")

    # Detenemos CodeCarbon tras la evaluación:
    emisiones = tracker.stop()

    print("Evaluacion PU sin Two-Step methods completada")
    print(f"Emisiones: {emisiones} kg CO2.\n")

    # Devolvemos los resultados PU sin Two-Step methods:
    return metricas


# Función de Entrenamiento y Evaluación PU con Two-Step methods:
def entr_y_eval_pu_con_ts(ruta_folds, ruta_folds_pu, num_folds, cl_positivas, metricas, nombre):
    print("\nEntrenamiento y Evaluacion PU con Two-Step methods:\n")

    # Iniciamos CodeCarbon:
    tracker = EmissionsTracker(
        project_name = f"{nombre}, PU con Two-Step",
        output_dir = RESULTADOS,
        output_file = "codecarbon.csv"
    )
    tracker.start()

    # Bucle principal:
    for fold_num in range(1, num_folds + 1):
        print(f"\nFold {fold_num}:")

        # Carga del dataset de test (solo uno de los folds, no PU):
        X_test, y_test = cargar_fold(ruta_folds / f"fold_{fold_num}.csv")

        # Se juntan el resto de folds para formar el dataset de entrenamiento, PU:
        indices_folds_entrenamiento = [i for i in range(1, num_folds + 1) if i != fold_num]
        X_train_pu, y_train_pu = juntar_folds_separados(ruta_folds_pu, indices_folds_entrenamiento)

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
        y_true_pu = np.isin(y_test, cl_positivas).astype(int)
        resultados_pu_con_ts = evaluacion_dataset(
            y_true_pu,
            y_pred,
            y_score
        )

        # Se añaden los resultados de las métricas a la lista:
        metricas.append(resultados_pu_con_ts)

        print(f"\nFold {fold_num} completado.")
    
    # Detenemos CodeCarbon tras la evaluación:
    emisiones = tracker.stop()

    print("Evaluacion PU con Two-Step methods completada")
    print(f"Emisiones: {emisiones} kg CO2.\n")

    # Devolvemos los resultados PU con Two-Step methods:
    return metricas


# Función principal, utilizada para el resto de datasets. Compara entrenamiento
# y evaluación no PU, PU sin Two-Step methods y PU con Two-Step methods:
def entrenamiento_y_eval(ruta_folds, ruta_folds_pu, num_folds, cl_positivas, porcentaje_pos, nombre, ruta_txt):
    # Se crean arrays para más tarde guardar los resultados:
    metricas_no_pu = []
    metricas_pu_sin_ts = []
    metricas_pu_con_ts = []

    # Información inicial:
    informacion_evaluacion(
        nombre,
        cl_positivas,
        porcentaje_pos,
        METRICA,
        MET_NEG_FIABLES,
        MET_APRENDIZAJE,
        True,
        ruta_txt
    )

    # Entrenamiento y evaluación no PU:
    metricas_no_pu = entr_y_eval_no_pu(
        ruta_folds,
        num_folds,
        cl_positivas,
        metricas_no_pu,
        nombre
    )

    # Entrenamiento y evaluación PU sin Two-Step Methods:
    metricas_pu_sin_ts = entr_y_eval_pu_sin_ts(
        ruta_folds,
        ruta_folds_pu,
        num_folds,
        cl_positivas,
        metricas_pu_sin_ts,
        nombre
    )

    # Entrenamiento y evaluación PU con Two-Step Methods:
    metricas_pu_con_ts = entr_y_eval_pu_con_ts(
        ruta_folds,
        ruta_folds_pu,
        num_folds,
        cl_positivas,
        metricas_pu_con_ts,
        nombre
    )

    # Se escriben las estadísticas finales en el archivo de guardado:
    escribir_resultados(metricas_no_pu, ruta_txt, "Resultados no PU")
    escribir_resultados(metricas_pu_sin_ts, ruta_txt, "Resultados PU sin Two-Step methods")
    escribir_resultados(metricas_pu_con_ts, ruta_txt, "Resultados PU con Two-Step methods")

    print("Evaluacion finalizada correctamente.")
    print(f"Informe guardado en: {ruta_txt}")