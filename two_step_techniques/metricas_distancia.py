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

# Función para comprobar si el nombre de una métrica es válido (si trabajamos con ella).
# Además, en el caso de la distancia de Manhattan ("Cityblock"), permite aceptar ambos nombres:
def comprobar_nombre_metrica(nombre):
        match nombre:
            case "euclidean":
                return "euclidean"
            case "cityblock" | "manhattan":
                return "cityblock"
            case "jaccard":
                return "jaccard"
            case "cosine":
                return "cosine"
            case "mahalanobis":
                return "mahalanobis"
            case "canberra":
                return "canberra"
            case "braycurtis":
                return "braycurtis"
            case _:
                raise ValueError(f"Metrica no valida: \"{nombre}\"")

# Función para, además de comprobar si la métrica es válida,
# convertir el string del nombre en una métrica de "scipy.spatial.distance":
def obtener_metrica(nombre):
    match nombre:
        case "euclidean":
            return euclidean
        case "cityblock" | "manhattan":
            return cityblock
        case "jaccard":
            return jaccard
        case "cosine":
            return cosine
        case "mahalanobis":
            return mahalanobis
        case "canberra":
            return canberra
        case "braycurtis":
            return braycurtis
        case _:
            raise ValueError(f"Metrica no valida: \"{nombre}\"")

# Función auxiliar para calcular la inversa de la Matriz de Covarianza "V".
# Se utiliza obligatoriamente con la distancia de Mahalanobis (parámetro "VI"):
def obtener_vi(X):
    V = np.cov(X.T)

    # Se utiliza la pseudoinversa, en caso de que la matriz no fuese invertible:
    VI = np.linalg.pinv(V)

    return VI