import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from config.paths_config import RESULTADOS
from config.train_config import *

from preprocesado.kfold import cargar_fold, juntar_folds_separados
from two_step_techniques.negativos_fiables import rocchio, knn, kmeans, kmedoids, crne
from two_step_techniques.aprendizaje import logistic_regression, random_forest, xgboost_lrn, mlp, cnn, crear_modelo_cnn
from evaluacion.evaluacion import informacion_evaluacion, evaluacion_dataset


# Función auxiliar para el preparado de datos. Transforma "X" a un array de numpy
# y la aplana (si fuese necesario) para poder calcular distancias correctamente:
def preparado_X(X, modelo):
    # "X" se transforma en array de numpy para una mayor eficiencia:
    X = np.asarray(X, dtype = np.float32)

    # Aplanamos "X" si fuese necesario (en imágenes):
    if modelo != "CNN" and X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    return X


# Función auxiliar para cargar los conjuntos de entrenamiento y test del mismo fold:
def cargar_par_fold(ruta_folds, fold_num, guardado_npy):
    X_train, y_train = cargar_fold(
        ruta_folds,
        fold_num,
        guardado_npy,
        es_train = True
    )

    X_test, y_test = cargar_fold(
        ruta_folds,
        fold_num,
        guardado_npy,
        es_test = True
    )

    return X_train, y_train, X_test, y_test


# Función auxiliar para la elección de un modelo de aprendizaje, sin Two-Step Methods:
def elegir_metodo_aprendizaje(metodo, X, semilla = None):
    # Se elige el método de aprendizaje:
    match metodo:
        case "Logistic Regression":
            modelo = LogisticRegression(random_state = semilla, max_iter = 1000)
        case "Random Forest":
            modelo = RandomForestClassifier(random_state = semilla)
        case "XGBoost":
            modelo = XGBClassifier(random_state = semilla)
        case "MLP":
            modelo = MLPClassifier(max_iter = 1000, random_state = semilla)
        case "CNN":
            modelo = crear_modelo_cnn(X)
        case _:
            raise ValueError(f"Modelo de aprendizaje no valido: \"{metodo}\"")
    
    return modelo


# Función auxiliar para calcular los pesos de las clases a partir del conjunto de entrenamiento:
def calcular_pesos(y_train):
    # Calculamos el número de ejemplos de cada clase:
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)

    # Calculamos el peso para cada clase:
    peso_pos = len(y_train) / (num_pos * 2)
    peso_neg = len(y_train) / (num_neg * 2)

    # Aplicamos los pesos:
    pesos = np.where(y_train == 1, peso_pos, peso_neg)

    return pesos


# Función auxiliar para obtener las predicciones del modelo tras el entrenamiento:
def realizar_prediccion(modelo, X_test, met_aprendizaje):
    # Con CNN, la última capa del modelo (la salida) ya devuelve directamente la probabilidad
    # de la clase positiva ("y_score"). Las predicciones ("y_pred") se obtienen utilizando esta:
    if met_aprendizaje == "CNN":
        y_score = modelo.predict(X_test).ravel()
        y_pred = (y_score >= 0.5).astype(int)

    # Con el resto de modelos, obtenemos "y_pred" e "y_score" por separado:
    else:
        y_pred = modelo.predict(X_test)
        y_score = modelo.predict_proba(X_test)[:, 1]

    return y_pred, y_score


# Función auxiliar para escribir en el archivo de resultados:
def escribir_resultados(metricas, num_folds, ruta_guardado, nombre):
    # Se abre el archivo de guardado:
    with open(ruta_guardado, "a") as f:
        f.write(f"{nombre}:\n")

        # En el caso extremo de que no se hayan obtenido resultados (por ejemplo, por no haber
        # encontrado ningún Negativo Fiable en el escenario de aprendizaje PU con Two-Step methods):
        if not metricas:
            f.write("No evaluable: no se han obtenido resultados.\n\n")
            return

        # En el caso de que alguno de los folds no haya obtenido resultados (por ejemplo, si ese fold
        # en concreto no hubiese encontrado ningún Negativo Fiable en el mismo escenario):
        if len(metricas) < num_folds:
            f.write(f"(Solo se han podido evaluar {len(metricas)} de los {num_folds} folds disponibles)\n")

        # Las métricas se convierten a un DataFrame de pandas:
        metricas_dataframe = pd.DataFrame(metricas)

        # Se calculan la media y la desviación típica de las métricas:
        metricas_media = metricas_dataframe.mean()
        metricas_dt = metricas_dataframe.std()

        # Se escriben los resultados:
        f.write(
            f"MEDIA: F1-Score = {metricas_media['F1-Score']:.3f}, "
            f"PR-AUC = {metricas_media['PR-AUC']:.3f}\n"
        )
        f.write(
            f"DESVIACION TIPICA: F1-Score = {metricas_dt['F1-Score']:.3f}, "
            f"PR-AUC = {metricas_dt['PR-AUC']:.3f}\n\n"
        )


# Función de Entrenamiento y Evaluación no PU:
def entr_y_eval_no_pu(ruta_folds, guardado_npy, num_folds, cl_positivas, metricas, nombre, sep_train_test = False):
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

        # Si cada fold tiene su separación entre entrenamiento y test:
        if sep_train_test:
            # Se cargan los folds correspondientes a ambos conjuntos:
            X_train, y_train, X_test, y_test = cargar_par_fold(ruta_folds, fold_num, guardado_npy)

        # En caso contrario:
        else:
            # Se carga solo uno de los folds, no PU, como conjunto de test:
            X_test, y_test = cargar_fold(ruta_folds, fold_num, guardado_npy)

            # El resto de folds se juntan para formar el conjunto de entrenamiento (en este caso, no PU):
            indices_folds_entrenamiento = [i for i in range(1, num_folds + 1) if i != fold_num]
            X_train, y_train = juntar_folds_separados(ruta_folds, indices_folds_entrenamiento, guardado_npy)

        # Se preparan las características (aplanando si es necesario) para el modelo de aprendizaje:
        X_train = preparado_X(X_train, MET_APRENDIZAJE)
        X_test = preparado_X(X_test, MET_APRENDIZAJE)

        # Si existiesen múltiples clases, las etiquetas se convierten a binarias.
        # Con "y_train", este paso ya se habrá realizado previamente en el resto de casos:
        y_train = np.isin(y_train, cl_positivas).astype(int)
        y_test = np.isin(y_test, cl_positivas).astype(int)

        # Se elige el modelo de aprendizaje:
        modelo_no_pu = elegir_metodo_aprendizaje(MET_APRENDIZAJE, X_train, SEMILLA_L)

        # Si la opción está activada, balanceamos los clases. Si no, entrenamos sin balancear:
        if BALANCEO_CLASES:
            pesos = calcular_pesos(y_train) # Calculamos los pesos de cada clase
            modelo_no_pu.fit(X_train, y_train, sample_weight = pesos)
        else:
            modelo_no_pu.fit(X_train, y_train)

        # Se realiza la predicción:
        y_pred_no_pu, y_score_no_pu = realizar_prediccion(modelo_no_pu, X_test, MET_APRENDIZAJE)

        # Se evalúa el modelo:
        resultados_no_pu = evaluacion_dataset(y_test, y_pred_no_pu, y_score_no_pu)

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
def entr_y_eval_pu_sin_ts(ruta_folds, ruta_folds_pu, guardado_npy, num_folds, cl_positivas, metricas, nombre, sep_train_test = False):
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

        # Si cada fold tiene su separación entre entrenamiento y test:
        if sep_train_test:
            # Se cargan los folds correspondientes a ambos conjuntos:
            X_train_pu, y_train_pu, X_test, y_test = cargar_par_fold(ruta_folds_pu, fold_num, guardado_npy)

        # En caso contrario:
        else:
            # Se carga solo uno de los folds, no PU, como conjunto de test:
            X_test, y_test = cargar_fold(ruta_folds, fold_num, guardado_npy)

            # El resto de folds se juntan para formar el conjunto de entrenamiento (en este caso, PU):
            indices_folds_entrenamiento = [i for i in range(1, num_folds + 1) if i != fold_num]
            X_train_pu, y_train_pu = juntar_folds_separados(ruta_folds_pu, indices_folds_entrenamiento, guardado_npy)

        # Se preparan las características (aplanando si es necesario) para el modelo de aprendizaje:
        X_train_pu = preparado_X(X_train_pu, MET_APRENDIZAJE)
        X_test = preparado_X(X_test, MET_APRENDIZAJE)

        # Se elige el modelo de aprendizaje:
        modelo_pu_sin_ts = elegir_metodo_aprendizaje(MET_APRENDIZAJE, X_train_pu, SEMILLA_L)

        # Si la opción está activada, balanceamos los clases. Si no, entrenamos sin balancear:
        if BALANCEO_CLASES:
            pesos = calcular_pesos(y_train_pu) # Calculamos los pesos de cada clase
            modelo_pu_sin_ts.fit(X_train_pu, y_train_pu, sample_weight = pesos)
        else:
            modelo_pu_sin_ts.fit(X_train_pu, y_train_pu)

        # Se realiza la predicción:
        y_pred_pu_sin_ts, y_score_pu_sin_ts = realizar_prediccion(modelo_pu_sin_ts, X_test, MET_APRENDIZAJE)

        # Se evalúa el modelo. Las etiquetas del conjunto de test se convierten a binarias previamente:
        y_test = np.isin(y_test, cl_positivas).astype(int)
        resultados_pu_sin_ts = evaluacion_dataset(y_test, y_pred_pu_sin_ts, y_score_pu_sin_ts)

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
def entr_y_eval_pu_con_ts(ruta_folds, ruta_folds_pu, guardado_npy, num_folds, cl_positivas, metricas, nombre, sep_train_test = False):
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

        # Si cada fold tiene su separación entre entrenamiento y test:
        if sep_train_test:
            # Se cargan los folds correspondientes a ambos conjuntos:
            X_train_pu, y_train_pu, X_test, y_test = cargar_par_fold(ruta_folds_pu, fold_num, guardado_npy)

        # En caso contrario:
        else:
            # Se carga solo uno de los folds, no PU, como conjunto de test:
            X_test, y_test = cargar_fold(ruta_folds, fold_num, guardado_npy)

            # El resto de folds se juntan para formar el conjunto de entrenamiento (en este caso, PU):
            indices_folds_entrenamiento = [i for i in range(1, num_folds + 1) if i != fold_num]
            X_train_pu, y_train_pu = juntar_folds_separados(ruta_folds_pu, indices_folds_entrenamiento, guardado_npy)

        # Creamos una versión plana de "X_train_pu", utilizada para poder obtener negativos fiables con cualquier
        # tipo de datos, pero pudiendo más tarde utilizar la versión sin aplanar (en el caso de CNN):
        X_train_pu_flt = preparado_X(X_train_pu, None)

        # Búsqueda de Negativos Fiables:
        match MET_NEG_FIABLES:
            case "Rocchio":
                indices_RN = rocchio(X_train_pu_flt, y_train_pu, METRICA)
            case "KNN":
                indices_RN = knn(X_train_pu_flt, y_train_pu, METRICA, K, PORCENTAJE_RN)
            case "KMeans":
                indices_RN = kmeans(X_train_pu_flt, y_train_pu, METRICA, K, PORCENTAJE_RN, SEMILLA_RN)
            case "KMedoids":
                indices_RN = kmedoids(X_train_pu_flt, y_train_pu, METRICA, K, PORCENTAJE_RN, SEMILLA_RN)
            case "CRNE":
                indices_RN = crne(X_train_pu_flt, y_train_pu, METRICA, K, SEMILLA_RN)
            case _:
                raise ValueError(f"Modelo de Negativos Fiables no valido: \"{MET_NEG_FIABLES}\"")

        # En el caso extremo de que no se hayan encontrado negativos fiables (posible en CRNE con valores de "k" bajos):
        if len(indices_RN) == 0:
            print("WARNING: no reliable negative examples have been identified. Fold has been skipped.")
            continue

        # Se preparan las características (aplanando si es necesario) para el modelo de aprendizaje:
        X_train_pu = preparado_X(X_train_pu, MET_APRENDIZAJE)
        X_test = preparado_X(X_test, MET_APRENDIZAJE)
        
        # Aprendizaje con Positivos y Negativos Fiables:
        match MET_APRENDIZAJE:
            case "Logistic Regression":
                modelo = logistic_regression(X_train_pu, y_train_pu, indices_RN, SEMILLA_L, BALANCEO_CLASES)
            case "Random Forest":
                modelo = random_forest(X_train_pu, y_train_pu, indices_RN, SEMILLA_L, BALANCEO_CLASES)
            case "XGBoost":
                modelo = xgboost_lrn(X_train_pu, y_train_pu, indices_RN, SEMILLA_L, BALANCEO_CLASES)
            case "MLP":
                modelo = mlp(X_train_pu, y_train_pu, indices_RN, SEMILLA_L, BALANCEO_CLASES)
            case "CNN":
                modelo = cnn(X_train_pu, y_train_pu, indices_RN, SEMILLA_L, BALANCEO_CLASES)
            case _:
                raise ValueError(f"Modelo de aprendizaje no valido: \"{MET_APRENDIZAJE}\"")

        # Se realiza la predicción:
        y_pred, y_score = realizar_prediccion(modelo, X_test, MET_APRENDIZAJE)

        # Se evalúa el modelo. Las etiquetas del conjunto de test se convierten a binarias previamente:
        y_test = np.isin(y_test, cl_positivas).astype(int)
        resultados_pu_con_ts = evaluacion_dataset(y_test, y_pred, y_score)

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
def entrenamiento_y_eval(ruta_folds, ruta_folds_pu, guardado_npy, num_folds, cl_positivas, porcentaje_pos, nombre, ruta_txt, solo_pu_con_ts = False):

    # Información inicial:
    informacion_evaluacion(
        nombre,
        cl_positivas,
        porcentaje_pos,
        MET_NEG_FIABLES,
        K, PORCENTAJE_RN,
        METRICA,
        MET_APRENDIZAJE,
        True,
        ruta_txt
    )

    # En caso de que se quieran entrenar y evaluar los tres escenarios:
    if not solo_pu_con_ts:

        # Entrenamiento y evaluación no PU. En cada escenario se crea un array para guardar los resultados:
        metricas_no_pu = []
        metricas_no_pu = entr_y_eval_no_pu(
            ruta_folds,
            guardado_npy,
            num_folds,
            cl_positivas,
            metricas_no_pu,
            nombre
        )

        # Se escriben las estadísticas finales en el archivo de guardado:
        escribir_resultados(metricas_no_pu, num_folds, ruta_txt, "Resultados no PU")

        # Entrenamiento y evaluación PU sin Two-Step Methods:
        metricas_pu_sin_ts = []
        metricas_pu_sin_ts = entr_y_eval_pu_sin_ts(
            ruta_folds,
            ruta_folds_pu,
            guardado_npy,
            num_folds,
            cl_positivas,
            metricas_pu_sin_ts,
            nombre
        )

        # Se escriben las estadísticas finales en el archivo de guardado:
        escribir_resultados(metricas_pu_sin_ts, num_folds, ruta_txt, "Resultados PU sin Two-Step methods")

    # Entrenamiento y evaluación PU con Two-Step Methods:
    metricas_pu_con_ts = []
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
    escribir_resultados(metricas_pu_con_ts, num_folds, ruta_txt, "Resultados PU con Two-Step methods")

    print("Evaluacion finalizada correctamente.")
    print(f"Informe guardado en: {ruta_txt}")


# Variación de la función principal, donde los folds vienen dados en k pares train-test.
# Para cada uno, se utiliza un par con un fold de entrenamiento y uno de test específico para este:
def entrenamiento_y_eval_train_test_separate(ruta_folds, ruta_folds_gen_pu, guardado_npy, num_folds, cl_positivas, porcentaje_pos, nombre, ruta_txt, solo_pu_con_ts = False):

    # Información inicial:
    informacion_evaluacion(
        nombre,
        cl_positivas,
        porcentaje_pos,
        MET_NEG_FIABLES,
        K, PORCENTAJE_RN,
        METRICA,
        MET_APRENDIZAJE,
        True,
        ruta_txt
    )

    # En caso de que se quieran entrenar y evaluar los tres escenarios:
    if not solo_pu_con_ts:

        # Entrenamiento y evaluación no PU. En cada escenario se crea un array para guardar los resultados:
        metricas_no_pu = []
        metricas_no_pu = entr_y_eval_no_pu(
            ruta_folds,
            guardado_npy,
            num_folds,
            cl_positivas,
            metricas_no_pu,
            nombre,
            True
        )

        # Se escriben las estadísticas finales en el archivo de guardado:
        escribir_resultados(metricas_no_pu, num_folds, ruta_txt, "Resultados no PU")

        # Entrenamiento y evaluación PU sin Two-Step Methods:
        metricas_pu_sin_ts = []
        metricas_pu_sin_ts = entr_y_eval_pu_sin_ts(
            ruta_folds,
            ruta_folds_gen_pu,
            guardado_npy,
            num_folds,
            cl_positivas,
            metricas_pu_sin_ts,
            nombre,
            True
        )

        # Se escriben las estadísticas finales en el archivo de guardado:
        escribir_resultados(metricas_pu_sin_ts, num_folds, ruta_txt, "Resultados PU sin Two-Step methods")

    # Entrenamiento y evaluación PU con Two-Step Methods:
    metricas_pu_con_ts = []
    metricas_pu_con_ts = entr_y_eval_pu_con_ts(
        ruta_folds,
        ruta_folds_gen_pu,
        guardado_npy,
        num_folds,
        cl_positivas,
        metricas_pu_con_ts,
        nombre,
        True
    )

    # Se escriben las estadísticas finales en el archivo de guardado:
    escribir_resultados(metricas_pu_con_ts, num_folds, ruta_txt, "Resultados PU con Two-Step methods")

    print("Evaluacion finalizada correctamente.")
    print(f"Informe guardado en: {ruta_txt}")