import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from two_step_techniques.negativos_fiables import preparado_datos


# Función auxiliar para construír el set de datos con el que se entrenará al modelo.
# Este se construye únicamente con los datos positivos (P) y los negativos fiables (RN):
def construir_dataset_entrenamiento(X, y, indices_RN, balanceo_clases = False):
    # Definimos el conjunto de positivos:
    P = X[y == 1]

    # Obtenemos los Negativos Fiables correspondientes a los índices dados:
    RN = X[indices_RN]

    # Creamos un nuevo conjunto de datos, juntando los positivos con los Negativos Fiables:
    X_train = np.vstack([P, RN])
    y_train = np.hstack([np.ones(len(P)), np.zeros(len(RN))])

    # Mezclamos este nuevo dataset para evitar sesgo:
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Si la opción está activada, calculamos el peso para cada clase:
    pesos = None
    if balanceo_clases:
        # Calculamos el número de ejemplos de cada clase:
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)

        # Después, calculamos los pesos:
        peso_pos = len(y_train) / (num_pos * 2)
        peso_neg = len(y_train) / (num_neg * 2)
        pesos = np.where(y_train == 1, peso_pos, peso_neg)

    return X_train, y_train, pesos


# Regresión Logística (Logistic Regression):
def logistic_regression(X, y, indices_RN, penalty = "l2", C = 1, semilla = None, balanceo_clases = False):
    # Preparamos los datos:
    X, y = preparado_datos(X, y)

    # Creamos el dataset de entrenamiento:
    X_train, y_train, pesos = construir_dataset_entrenamiento(X, y, indices_RN, balanceo_clases)

    # Creamos el clasificador y lo entrenamos:
    modelo = LogisticRegression(
        penalty = penalty,      # Regularización
        C = C,                  # Inverso de regularización
        random_state = semilla,
        max_iter = 1000
    )
    modelo.fit(X_train, y_train, sample_weight = pesos)

    print("\nAprendizaje con Regresión Logística completado.")
    return modelo


# Random Forest:
def random_forest(X, y, indices_RN, n_estimators = 10, criterion = "gini", max_depth = 50, semilla = None, balanceo_clases = False):
    # Preparamos los datos:
    X, y = preparado_datos(X, y)

    # Creamos el dataset de entrenamiento:
    X_train, y_train, pesos = construir_dataset_entrenamiento(X, y, indices_RN, balanceo_clases)

    # Creamos el clasificador y lo entrenamos:
    modelo = RandomForestClassifier(
        n_estimators = n_estimators,    # Número de árboles
        criterion = criterion,          # Función para cortes en cada árbol
        max_depth = max_depth,          # Profundidad del árbol
        random_state = semilla
    )
    modelo.fit(X_train, y_train, sample_weight = pesos)

    print("\nAprendizaje con Random Forest completado.")
    return modelo


# XGBoost (eXtreme Gradient Boosting):
def xgboost_lrn(X, y, indices_RN, semilla = None, balanceo_clases = False):
    # Preparamos los datos:
    X, y = preparado_datos(X, y)

    # Creamos el dataset de entrenamiento:
    X_train, y_train, pesos = construir_dataset_entrenamiento(X, y, indices_RN, balanceo_clases)

    # Creamos el clasificador y lo entrenamos:
    modelo = XGBClassifier(random_state = semilla)
    modelo.fit(X_train, y_train, sample_weight = pesos)

    print("\nAprendizaje con XGBoost completado.")
    return modelo


# Perceptrón Multicapa (MLP):
def mlp(X, y, indices_RN, hidden_layer_sizes = (10, 10), activation = "tanh", semilla = None, balanceo_clases = False):
    # Preparamos los datos:
    X, y = preparado_datos(X, y)

    # Creamos el dataset de entrenamiento:
    X_train, y_train, pesos = construir_dataset_entrenamiento(X, y, indices_RN, balanceo_clases)

    # Creamos el clasificador y lo entrenamos:
    modelo = MLPClassifier(
        hidden_layer_sizes = hidden_layer_sizes,    # Arquitectura (capas ocultas)
        activation = activation,                    # Función no lineal
        max_iter = 1000,
        random_state = semilla
    )
    modelo.fit(X_train, y_train, sample_weight = pesos)

    print("\nAprendizaje con Perceptron Multicapa (MLP) completado.")
    return modelo


# Red Neuronal Convolucional (CNN):
def cnn(X, y, indices_RN, optimizer = "adam", loss = "binary_crossentropy", semilla = None, balanceo_clases = False, aprendizaje_sin_pu = False):
    # "X" e "y" se transforman en arrays de numpy para una mayor eficiencia:
    X = np.asarray(X, dtype = np.float32)
    y = np.asarray(y)

    # En caso de estar utilizando CNN sin aprendizaje PU:
    if aprendizaje_sin_pu:
        X_train = X.copy()
        y_train = y.copy()
        pesos = None

    # En caso contrario:
    else:
        # Creamos el dataset de entrenamiento. Los datos no se aplanan de antemano:
        X_train, y_train, pesos = construir_dataset_entrenamiento(X, y, indices_RN, balanceo_clases)

    # Creamos el modelo de CNN:
    modelo = Sequential([
        Input(shape = X_train.shape[1:]),
        Conv2D(32, (3, 3), activation = "relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation = "relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation = "relu"),
        Dense(1, activation = "sigmoid")
    ])

    # Configuramos el modelo:
    modelo.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = ["accuracy"]
    )

    # Entrenamos el clasificador:
    modelo.fit(X_train, y_train, sample_weight = pesos)

    print("\nAprendizaje con Red Neuronal Convolucional (CNN) completado.")
    return modelo