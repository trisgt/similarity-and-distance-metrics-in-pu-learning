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


# Diccionario de métricas de distancia utilizadas en el proyecto, relacionando
# los nombres con su correspondiente métrica de "scipy.spatial.distance":
METRICAS_DISTANCIA = {
    "euclidean": euclidean,
    "cityblock": cityblock,
    "manhattan": cityblock, # "manhattan" se considera un alias de "cityblock"
    "jaccard": jaccard,
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