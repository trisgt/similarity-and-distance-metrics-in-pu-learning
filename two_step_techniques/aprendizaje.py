import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# Función auxiliar para construír el set de datos con el que se entrenará al modelo.
# Este se construye únicamente con los datos positivos (P) y los negativos fiables (RN):
def construir_dataset_entrenamiento(X, y, RN):
    # Primero, definimos el conjunto de positivos:
    P = X[y == 1]

    # Juntamos los positivos en las columnas izquierdas y los RN en las derechas:
    X_train = np.vstack([P, RN])
    y_train = np.hstack([np.ones(len(P)), np.zeros(len(RN))])

    # Mezclamos el nuevo dataset para evitar sesgo:
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    return X_train, y_train


# Logistic Regression:
def logistic_regression(X, y, RN, penalty = "l2", C = 1, semilla = None):
    # Creamos el dataset:
    X_train, y_train = construir_dataset_entrenamiento(X, y, RN)

    # Creamos el clasificador y lo entrenamos:
    modelo = LogisticRegression(
        penalty = penalty,      # Regularización
        C = C,                  # Inverso de regularización
        random_state = semilla,
        max_iter = 1000
    )
    modelo.fit(X_train, y_train)

    print("\nAprendizaje con Logistic Regression completado.")
    return modelo


# Random Forest:
def random_forest(X, y, RN, n_estimators = 10, criterion = "gini", max_depth = 50, semilla = None):
    # Creamos el dataset:
    X_train, y_train = construir_dataset_entrenamiento(X, y, RN)

    # Creamos el clasificador y lo entrenamos:
    modelo = RandomForestClassifier(
        n_estimators = n_estimators,    # Número de árboles
        criterion = criterion,          # Función para cortes en cada árbol
        max_depth = max_depth,          # Profundidad del árbol
        random_state = semilla
    )
    modelo.fit(X_train, y_train)

    print("\nAprendizaje con Random Forest completado.")
    return modelo


# XGBoost:
def xgboost_lrn(X, y, RN, semilla = None):
    # Creamos el dataset:
    X_train, y_train = construir_dataset_entrenamiento(X, y, RN)

    # Creamos el clasificador y lo entrenamos:
    modelo = XGBClassifier(random_state = semilla)
    modelo.fit(X_train, y_train)

    print("\nAprendizaje con XGBoost completado.")
    return modelo


# Perceptrón Multicapa (MLP):
def mlp(X, y, RN, hidden_layer_sizes = (10, 10), activation = "tanh", semilla = None):
    # Creamos el dataset:
    X_train, y_train = construir_dataset_entrenamiento(X, y, RN)

    # Creamos el clasificador y lo entrenamos:
    modelo = MLPClassifier(
        hidden_layer_sizes = hidden_layer_sizes,    # Arquitectura (capas ocultas)
        activation = activation,                    # Función no lineal
        max_iter = 1000,
        random_state = semilla
    )
    modelo.fit(X_train, y_train)

    print("\nAprendizaje con Perceptron Multicapa (MLP) completado.")
    return modelo


# Para CNN, necesito que X sea un tensor (num_imagenes, alto, ancho, canales).
# Usar PyTorch, más eficiente que Tensorflow según un estudio
# Hacer un preprocesado especial

# def cnn(X, y, RN, semilla = 1):
# (...)