import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

from config.paths_config import RESULTADOS
from config.train_config import *

from preprocesado.kfold import cargar_fold, juntar_folds_separados
from two_step_techniques.negativos_fiables import rocchio, knn, kmeans, kmedoids, crne
from two_step_techniques.aprendizaje import logistic_regression, random_forest, xgboost_lrn, mlp, cnn
from evaluacion.evaluacion import informacion_evaluacion, evaluacion_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# Función auxiliar para la elección de un modelo de aprendizaje (sin Two-Step). No se encarga de CNN:
def elegir_metodo_aprendizaje(
        metodo,
        penalty = "l2",
        c = 1.0,
        n_estimators = 100,
        criterion = "gini",
        max_depth = None,
        hidden_layer_sizes = (100, ),
        activation = "relu",
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

# Función auxiliar para el preparado de datos. Transforma "X" a array de numpy
# y la aplana (si fuese necesario) para poder calcular distancias correctamente:
def preparado_X(X, modelo):
    # "X" se transforma en array de numpy para una mayor eficiencia:
    X = np.asarray(X, dtype = np.float32)

    # Si fuese necesario (por ejemplo, con imágenes), la aplanamos:
    if modelo != "CNN" and X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    return X

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
def entr_y_eval_no_pu(ruta_folds, guardado_npy, num_folds, cl_positivas, metricas, nombre):
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
        X_test, y_test = cargar_fold(ruta_folds, fold_num, guardado_npy)

        # Se juntan el resto de folds para formar el dataset de entrenamiento, no PU:
        indices_folds_entrenamiento = [i for i in range(1, num_folds + 1) if i != fold_num]
        X_train, y_train = juntar_folds_separados(ruta_folds, indices_folds_entrenamiento, guardado_npy)

        # Se convierten las etiquetas multiclase a binarias:
        y_train = np.isin(y_train, cl_positivas).astype(int)
        y_test = np.isin(y_test, cl_positivas).astype(int)

        # Se preparan los datos para el modelo de aprendizaje:
        X_train = preparado_X(X_train, MET_APRENDIZAJE)
        X_test = preparado_X(X_test, MET_APRENDIZAJE)

        # Se elige el modelo no PU:
        if MET_APRENDIZAJE == "CNN":
            # Con CNN, el entrenamiento se hace directamente:
            modelo_no_pu = cnn(
                X_train,
                y_train,
                None,
                OPTIMIZER,
                LOSS,
                SEMILLA_L,
                BALANCEO_CLASES,
                True)
        else:
            modelo_no_pu = elegir_metodo_aprendizaje(
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
                pesos = calcular_pesos(y_train) # Calculamos los pesos de cada clase
                modelo_no_pu.fit(X_train, y_train, sample_weight = pesos)
            else:
                modelo_no_pu.fit(X_train, y_train)

        # Obtenemos también las probabilidades que el modelo asigna a la clase positiva:
        if MET_APRENDIZAJE == "CNN":
            y_score_no_pu = modelo_no_pu.predict(X_test).flatten()
            y_pred_no_pu = (y_score_no_pu >= 0.5).astype(int)
        else:
            # El modelo entrenado se utiliza para predecir las clases del conjunto de test:
            y_pred_no_pu = modelo_no_pu.predict(X_test)
            y_score_no_pu = modelo_no_pu.predict_proba(X_test)[:, 1]

        # Se evalúa el modelo:
        resultados_no_pu = evaluacion_dataset(
            y_test,
            y_pred_no_pu,
            y_score_no_pu
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
def entr_y_eval_pu_sin_ts(ruta_folds, ruta_folds_pu, guardado_npy, num_folds, cl_positivas, metricas, nombre):
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
        X_test, y_test = cargar_fold(ruta_folds, fold_num, guardado_npy)

        # Se juntan el resto de folds para formar el dataset de entrenamiento, en este caso PU:
        indices_folds_entrenamiento = [i for i in range(1, num_folds + 1) if i != fold_num]
        X_train_pu, y_train_pu = juntar_folds_separados(ruta_folds_pu, indices_folds_entrenamiento, guardado_npy)

        # Se preparan los datos para el modelo de aprendizaje:
        X_train_pu = preparado_X(X_train_pu, MET_APRENDIZAJE)
        X_test = preparado_X(X_test, MET_APRENDIZAJE)

        # Se elige el modelo PU sin Two-Step methods:
        if MET_APRENDIZAJE == "CNN":
            # Con CNN, el entrenamiento se hace directamente:
            modelo_pu_sin_ts = cnn(
                X_train_pu,
                y_train_pu,
                None,
                OPTIMIZER,
                LOSS,
                SEMILLA_L,
                BALANCEO_CLASES,
                True)
        else:
            modelo_pu_sin_ts = elegir_metodo_aprendizaje(
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
                modelo_pu_sin_ts.fit(X_train_pu, y_train_pu, sample_weight = pesos)
            else:
                modelo_pu_sin_ts.fit(X_train_pu, y_train_pu)

        # Obtenemos también las probabilidades que el modelo asigna a la clase positiva:
        if MET_APRENDIZAJE == "CNN":
            y_score_pu_sin_ts = modelo_pu_sin_ts.predict(X_test).flatten()
            y_pred_pu_sin_ts = (y_score_pu_sin_ts >= 0.5).astype(int)
        else:
            # El modelo entrenado se utiliza para predecir las clases del conjunto de test:
            y_pred_pu_sin_ts = modelo_pu_sin_ts.predict(X_test)
            y_score_pu_sin_ts = modelo_pu_sin_ts.predict_proba(X_test)[:, 1]

        # Se evalúa el modelo:
        y_true_pu = np.isin(y_test, cl_positivas).astype(int)
        resultados_pu_sin_ts = evaluacion_dataset(
            y_true_pu,
            y_pred_pu_sin_ts,
            y_score_pu_sin_ts
        )

        # Se añaden los resultados de las métricas a la lista:
        metricas.append(resultados_pu_sin_ts)

        print(f"Fold {fold_num} completado.")

    # Detenemos CodeCarbon tras la evaluación:
    emisiones = tracker.stop()

    print("Evaluacion PU sin Two-Step methods completada")
    print(f"Emisiones: {emisiones} kg CO2.\n")

    # Devolvemos los resultados PU sin Two-Step methods:
    return metricas


# Función de Entrenamiento y Evaluación PU con Two-Step methods:
def entr_y_eval_pu_con_ts(ruta_folds, ruta_folds_pu, guardado_npy, num_folds, cl_positivas, metricas, nombre):
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
        X_test, y_test = cargar_fold(ruta_folds, fold_num, guardado_npy)

        # Se juntan el resto de folds para formar el dataset de entrenamiento, PU:
        indices_folds_entrenamiento = [i for i in range(1, num_folds + 1) if i != fold_num]
        X_train_pu, y_train_pu = juntar_folds_separados(ruta_folds_pu, indices_folds_entrenamiento, guardado_npy)

        # Se crea una versión plana de X_train_pu, para poder obtener negativos fiables con datos de imágenes:
        X_train_pu_flat = preparado_X(X_train_pu, None)

        # Búsqueda de Negativos Fiables:
        match MET_NEG_FIABLES:
            case "Rocchio":
                indices_RN = rocchio(X_train_pu_flat, y_train_pu, METRICA)
            case "KNN":
                indices_RN = knn(X_train_pu_flat, y_train_pu, METRICA, K, PORCENTAJE_RN)
            case "KMeans":
                indices_RN = kmeans(X_train_pu_flat, y_train_pu, METRICA, K, PORCENTAJE_RN, SEMILLA_RN)
            case "KMedoids":
                indices_RN = kmedoids(X_train_pu_flat, y_train_pu, METRICA, K, PORCENTAJE_RN, SEMILLA_RN)
            case "CRNE":
                indices_RN = crne(X_train_pu_flat, y_train_pu, METRICA, K, SEMILLA_RN)
            case _:
                raise ValueError(f"Modelo de Negativos Fiables no valido: \"{MET_NEG_FIABLES}\"")

        # Se preparan los datos para el modelo de aprendizaje:
        X_train_pu = preparado_X(X_train_pu, MET_APRENDIZAJE)
        X_test = preparado_X(X_test, MET_APRENDIZAJE)
        
        # Aprendizaje con Positivos y Negativos Fiables:
        match MET_APRENDIZAJE:
            case "Logistic Regression":
                modelo = logistic_regression(X_train_pu, y_train_pu, indices_RN, PENALTY, C, SEMILLA_L, BALANCEO_CLASES)
            case "Random Forest":
                modelo = random_forest(X_train_pu, y_train_pu, indices_RN, N_ESTIMATORS, CRITERION, MAX_DEPTH, SEMILLA_L, BALANCEO_CLASES)
            case "XGBoost":
                modelo = xgboost_lrn(X_train_pu, y_train_pu, indices_RN, SEMILLA_L, BALANCEO_CLASES)
            case "MLP":
                modelo = mlp(X_train_pu, y_train_pu, indices_RN, HIDDEN_LAYER_SIZES, ACTIVATION, SEMILLA_L, BALANCEO_CLASES)
            case "CNN":
                modelo = cnn(X_train_pu, y_train_pu, indices_RN, OPTIMIZER, LOSS, SEMILLA_L, BALANCEO_CLASES, False)
            case _:
                raise ValueError(f"Modelo de aprendizaje no valido: \"{MET_APRENDIZAJE}\"")

        # Obtenemos también las probabilidades que el modelo asigna a la clase positiva:
        if MET_APRENDIZAJE == "CNN":
            y_score = modelo.predict(X_test).flatten()
            y_pred = (y_score >= 0.5).astype(int)
        else:
            # El modelo entrenado se utiliza para predecir las clases del conjunto de test:
            y_pred = modelo.predict(X_test)
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
def entrenamiento_y_eval(ruta_folds, ruta_folds_pu, guardado_npy, num_folds, cl_positivas, porcentaje_pos, nombre, ruta_txt):
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
        guardado_npy,
        num_folds,
        cl_positivas,
        metricas_no_pu,
        nombre
    )

    # Entrenamiento y evaluación PU sin Two-Step Methods:
    metricas_pu_sin_ts = entr_y_eval_pu_sin_ts(
        ruta_folds,
        ruta_folds_pu,
        guardado_npy,
        num_folds,
        cl_positivas,
        metricas_pu_sin_ts,
        nombre
    )

    # Entrenamiento y evaluación PU con Two-Step Methods:
    metricas_pu_con_ts = entr_y_eval_pu_con_ts(
        ruta_folds,
        ruta_folds_pu,
        guardado_npy,
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