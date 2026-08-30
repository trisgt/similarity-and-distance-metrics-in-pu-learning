import numpy as np
from scipy.spatial.distance import(
    euclidean,      # Distancia Euclídea
    cityblock,      # Distancia de Manhattan
    jaccard,        # Distancia de Jaccard
    cosine,         # Distancia de Coseno
    mahalanobis,    # Distancia de Mahalanobis
    canberra,       # Distancia de Canberra
    braycurtis      # Distancia de Bray-Curtis
)

'''
# Función auxiliar para calcular la distancia de Jaccard generalizada para valores reales:
def real_val_jaccard(x, y):
    # Calculamos la magnitud de cada componente:
    abs_x = np.abs(x)
    abs_y = np.abs(y)

    # Calculamos el numerador y denominador del índice de Jaccard:
    num = np.sum(np.sign(x * y) * np.minimum(abs_x, abs_y))
    den = np.sum(np.maximum(abs_x, abs_y))

    # En caso de que el denominador sea nulo, para evitar división entre 0:
    if den == 0:
        return 0.0

    return 1.0 - (num / den)
'''

# Función auxiliar para calcular la distancia de Jaccard generalizada para valores continuos:
def gen_jaccard(x, y):
    # Calculamos el numerador y denominador del índice de Jaccard:
    num = np.sum(np.minimum(x, y))
    den = np.sum(np.maximum(x, y))

    # En caso de que el denominador sea nulo, para evitar división entre 0:
    if den == 0:
        return 0.0

    # Convertimos el índice de similitud en distancia:
    return 1.0 - (num / den)


# Diccionario de métricas de distancia utilizadas en el proyecto, relacionando
# los nombres con su correspondiente métrica de "scipy.spatial.distance":
METRICAS_DISTANCIA = {
    "euclidean": euclidean,
    "cityblock": cityblock,
    "manhattan": cityblock, # "manhattan" se considera un alias de "cityblock"
    "jaccard": gen_jaccard, # Con Jaccard se utiliza la distancia generalizada
    "cosine": cosine,
    "mahalanobis": mahalanobis,
    "canberra": canberra,
    "braycurtis": braycurtis,
}

# Función para comprobar si el nombre de una métrica es válido
# (es decir, si se corresponde con una de las utilizadas en el proyecto):
def comprobar_nombre_metrica(nombre):
        if nombre not in METRICAS_DISTANCIA:
            raise ValueError(f"Metrica no valida: \"{nombre}\"")

        return nombre

# Función para devolver la función de SciPy correspondiente a una métrica:
def obtener_metrica(nombre):
    try:
        return METRICAS_DISTANCIA[nombre]
    except KeyError:
        raise ValueError(f"Metrica no valida: \"{nombre}\"")


# Función auxiliar para calcular la inversa de la Matriz de Covarianza "V".
# Se utiliza obligatoriamente con la distancia de Mahalanobis (parámetro "VI"):
def obtener_vi(X):
    V = np.cov(X, rowvar = False)

    # Se utiliza la pseudoinversa, en caso de que la matriz no fuese invertible:
    VI = np.linalg.pinv(V)

    return VI