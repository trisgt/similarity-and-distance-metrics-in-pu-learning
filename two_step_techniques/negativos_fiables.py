import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from two_step_techniques.metricas_distancia import comprobar_nombre_metrica, obtener_metrica, obtener_vi


# Función auxiliar para el preparado de datos. Transforma "X" e "y" a arrays de numpy
# y aplana "X" (si fuese necesario) para poder calcular distancias correctamente:
def preparado_datos(X, y, aplanar = True):
    # "X" e "y" se transforman en arrays de numpy para una mayor eficiencia:
    X = np.asarray(X, dtype = np.float32)
    y = np.asarray(y)

    # Si se indicase la opción y fuese necesario (en imágenes), aplanamos "X":
    if aplanar:
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

    return X, y


# Función auxiliar para comprobar que "k" no supere al número de muestras disponibles:
def comprobar_k(k, num_muestras):
    if k > num_muestras:
        raise ValueError(f"El valor de k ({k}) no puede ser mayor que el numero de muestras disponibles ({num_muestras})")


# Método de Rocchio. En cada método se utiliza un dataset ya convertido a PU ("engineered"):
def rocchio(X, y, nombre_metrica):
    # Comprobamos que la métrica sea válida:
    nombre_metrica = comprobar_nombre_metrica(nombre_metrica)

    # Preparamos los datos:
    X, y = preparado_datos(X, y)

    # Obtenemos los índices de las muestras no etiquetadas:
    indices_U = np.where(y == 0)[0]

    # Separamos las muestras Positivas y las No Etiquetadas:
    P = X[y == 1]
    U = X[y == 0]

    # Cada clase se representa mediante un prototpio de Rocchio (centroide medio).
    # Se calculan el de los Positivos y el de los No Etiquetados:
    prototipo_p = np.mean(P, axis = 0).reshape(1, -1)
    prototipo_u = np.mean(U, axis = 0).reshape(1, -1)

    # Calculamos las distancias de todos los No Etiquetados a ambos prototpios:
    if nombre_metrica == "mahalanobis": # Para Mahalanobis se necesita utilizar VI
        VI = obtener_vi(X)
        dist_prot_p = pairwise_distances(U, prototipo_p, nombre_metrica, VI = VI).ravel()
        dist_prot_u = pairwise_distances(U, prototipo_u, nombre_metrica, VI = VI).ravel()
    else:
        dist_prot_p = pairwise_distances(U, prototipo_p, nombre_metrica).ravel()
        dist_prot_u = pairwise_distances(U, prototipo_u, nombre_metrica).ravel()

    # Obtenemos los índices de los negativos fiables. Estos se corresponderán con
    # las muestras que estén mas cerca del prototipo de No Etiquetados que del de Positivos:
    indices_RN = indices_U[dist_prot_u < dist_prot_p]

    print("\nIdentificacion de Negativos Fiables con Rocchio completada:")
    print(f"\tPositivos: {len(P)}")
    print(f"\tNo Etiquetados: {len(U)}")
    print(f"\tNegativos Fiables: {len(indices_RN)}")

    return indices_RN


# KNN (K-Nearest Neighbours):
def knn(X, y, nombre_metrica, k, porcentaje_rn):
    # Comprobamos que la métrica sea válida:
    nombre_metrica = comprobar_nombre_metrica(nombre_metrica)

    # Preparamos los datos:
    X, y = preparado_datos(X, y)

    # Obtenemos los índices de las muestras no etiquetadas:
    indices_U = np.where(y == 0)[0]

    # Separamos las muestras Positivas y las No Etiquetadas:
    P = X[y == 1]
    U = X[y == 0]

    # Comprobamos que "k" no sea mayor al número de muestras positivas:
    comprobar_k(k, len(P))

    # Creamos un modelo de KNN:
    if nombre_metrica == "mahalanobis":
        VI = obtener_vi(X)
        modelo_knn = NearestNeighbors(
            n_neighbors = k,
            metric = nombre_metrica,
            metric_params = {"VI": VI})
    else:
        modelo_knn = NearestNeighbors(
            n_neighbors = k,
            metric = nombre_metrica)

    # Con "fit", se crea un mapa de las muestras positivas, que el modelo podrá utilizar:
    modelo_knn.fit(P)

    # Para cada No Etiquetado, obtenemos su distancia media a los k vecinos positivos más cercanos:
    distancias, _ = modelo_knn.kneighbors(U) # Sólo nos interesan las distancias obtenidas, no los índices de los puntos
    dist_medias = np.mean(distancias, axis = 1)

    # Se deciden cuáles son los No Etiquetados más alejados de los positivos (Negativos Fiables).
    # Para separarlos, se utiliza el porcentaje de negativos fiables:
    threshold = np.quantile(dist_medias, 1 - porcentaje_rn)
    indices_RN = indices_U[dist_medias >= threshold]

    print("\nIdentificacion de Negativos Fiables con KNN completada:")
    print(f"\tPositivos: {len(P)}")
    print(f"\tNo Etiquetados: {len(U)}")
    print(f"\tNegativos Fiables: {len(indices_RN)}")

    return indices_RN


# K-Means:
def kmeans(X, y, nombre_metrica, k, porcentaje_rn, semilla = 1):
    # Comprobamos que la métrica sea válida:
    nombre_metrica = comprobar_nombre_metrica(nombre_metrica)

    # Preparamos los datos:
    X, y = preparado_datos(X, y)

    # Obtenemos los índices de las muestras no etiquetadas:
    indices_U = np.where(y == 0)[0]

    # Separamos las muestras Positivas y las No Etiquetadas:
    P = X[y == 1]
    U = X[y == 0]

    # Comprobamos que "k" no sea mayor al número de muestras no etiquetadas:
    comprobar_k(k, len(U))

    # Creamos un modelo de K-Means (con distancia Euclídea). Con "fit_predict" se ejecuta
    # el algoritmo de K-Means sobre los No Etiquetados, obteniendo k clusters que ya han convergido:
    modelo_kmeans = KMeans(n_clusters = k, random_state = semilla)
    clusters_u = modelo_kmeans.fit_predict(U)

    # Obtenemos los centroides de cada cluster de los No Etiquetados, y el centroide (prototipo) positivo:
    centroides_u = modelo_kmeans.cluster_centers_
    prototipo_p = np.mean(P, axis = 0).reshape(1, -1)

    # Calculamos la distancia de cada centroide al prototipo positivo:
    VI = None
    if nombre_metrica == "mahalanobis":
        VI = obtener_vi(X)
        dist_centroides = pairwise_distances(centroides_u, prototipo_p, nombre_metrica, VI = VI).ravel()
    else:
        dist_centroides = pairwise_distances(centroides_u, prototipo_p, nombre_metrica).ravel()

    # Ordenamos los centroides por distancia, de más lejanos a más cercanos al positivo:
    orden_centroides = np.argsort(dist_centroides)[::-1]

    # Seleccionamos la mitad de estos clústeres como candidatos a negativos:
    num_clusters_neg = max(1, k // 2)
    clusters_neg = orden_centroides[:num_clusters_neg]

    # Ahora, seleccionamos las muestras individuales candidatas dentro de estos clústeres.
    # Estas son las que más alejadas se encuentran del centroide positivo:
    mask_candidatos = np.isin(clusters_u, clusters_neg)
    candidatos = U[mask_candidatos]
    indices_candidatos = indices_U[mask_candidatos]

    # Calculamos la distancia de cada candidato al prototipo positivo:
    if nombre_metrica == "mahalanobis":
        dist_candidatos = pairwise_distances(candidatos, prototipo_p, nombre_metrica, VI = VI).ravel()
    else:
        dist_candidatos = pairwise_distances(candidatos, prototipo_p, nombre_metrica).ravel()

    # Ordenamos los candidatos por distancia, de más lejanos a más cercanos al positivo:
    orden_candidatos = np.argsort(dist_candidatos)[::-1]

    # Seleccionamos el porcentaje seleccionado de los candidatos como Negativos Fiables:
    num_rn = min(int(len(U) * porcentaje_rn), len(candidatos))
    indices_RN = indices_candidatos[orden_candidatos[:num_rn]]

    print("\nIdentificacion de Negativos Fiables con K-Means completada:")
    print(f"\tPositivos: {len(P)}")
    print(f"\tNo Etiquetados: {len(U)}")
    print(f"\tNegativos Fiables: {len(indices_RN)}")

    return indices_RN


# K-Medoids:
def kmedoids(X, y, nombre_metrica, k, porcentaje_rn, semilla = 1):
    # Comprobamos que la métrica sea válida:
    nombre_metrica = comprobar_nombre_metrica(nombre_metrica)

    # Preparamos los datos:
    X, y = preparado_datos(X, y)

    # Obtenemos los índices de las muestras no etiquetadas:
    indices_U = np.where(y == 0)[0]

    # Separamos las muestras Positivas y las No Etiquetadas:
    P = X[y == 1]
    U = X[y == 0]

    # Comprobamos que "k" no sea mayor al número de muestras no etiquetadas:
    comprobar_k(k, len(U))

    # Creamos un modelo de K-Medoids. Con "fit_predict" se ejecuta el algoritmo de K-Medoids sobre
    # los No Etiquetados, obteniendo k clusters que ya han convergido. A diferencia de K-Means,
    # K-Medoids sí que puede utilizar la métrica especificada durante el proceso de clustering:
    VI = None
    if nombre_metrica == "mahalanobis":
        VI = obtener_vi(X)
        matriz_distancias = pairwise_distances(U, metric = nombre_metrica, VI = VI)
        modelo_kmedoids = KMedoids(n_clusters = k, metric = "precomputed", random_state = semilla)
        clusters_u = modelo_kmedoids.fit_predict(matriz_distancias)
    else:
        modelo_kmedoids = KMedoids(n_clusters = k, metric = nombre_metrica, random_state = semilla)
        clusters_u = modelo_kmedoids.fit_predict(U)

    # Obtenemos los centroides (medoides) de cada cluster de los No Etiquetados, y el centroide (prototipo) positivo.
    # La diferencia con K-Means es que los medoides son puntos reales del dataset, y no medias:
    medoides_u = U[modelo_kmedoids.medoid_indices_]
    prototipo_p = np.mean(P, axis = 0).reshape(1, -1)

    # Calculamos la distancia de cada medoide al prototipo positivo:
    if nombre_metrica == "mahalanobis":
        dist_medoides = pairwise_distances(medoides_u, prototipo_p, nombre_metrica, VI = VI).ravel()
    else:
        dist_medoides = pairwise_distances(medoides_u, prototipo_p, nombre_metrica).ravel()

    # Ordenamos los medoides por distancia, de más lejanos a más cercanos al positivo:
    orden_medoides = np.argsort(dist_medoides)[::-1]

    # Seleccionamos la mitad de estos clústeres como candidatos a negativos:
    num_clusters_neg = max(1, k // 2)
    clusters_neg = orden_medoides[:num_clusters_neg]

    # Ahora, seleccionamos las muestras individuales candidatas dentro de estos clústeres.
    # Estas son las que más alejadas se encuentran del centroide positivo:
    mask_candidatos = np.isin(clusters_u, clusters_neg)
    candidatos = U[mask_candidatos]
    indices_candidatos = indices_U[mask_candidatos]

    # Calculamos la distancia de cada candidato al prototipo positivo:
    if nombre_metrica == "mahalanobis":
        dist_candidatos = pairwise_distances(candidatos, prototipo_p, nombre_metrica, VI = VI).ravel()
    else:
        dist_candidatos = pairwise_distances(candidatos, prototipo_p, nombre_metrica).ravel()

    # Ordenamos los candidatos por distancia, de más lejanos a más cercanos al positivo:
    orden_candidatos = np.argsort(dist_candidatos)[::-1]

    # Seleccionamos el porcentaje seleccionado de los candidatos como Negativos Fiables:
    num_rn = min(int(len(U) * porcentaje_rn), len(candidatos))
    indices_RN = indices_candidatos[orden_candidatos[:num_rn]]

    print("\nIdentificacion de Negativos Fiables con K-Medoids completada:")
    print(f"\tPositivos: {len(P)}")
    print(f"\tNo Etiquetados: {len(U)}")
    print(f"\tNegativos Fiables: {len(indices_RN)}")

    return indices_RN


# CRNE (C-CRNE):
def crne(X, y, nombre_metrica, k, semilla = 1):
    # Comprobamos que la métrica sea válida:
    nombre_metrica = comprobar_nombre_metrica(nombre_metrica)

    # Preparamos los datos:
    X, y = preparado_datos(X, y)

    # Comprobamos que "k" no sea mayor al número de muestras:
    comprobar_k(k, len(X))

    # Creamos un modelo de K-Medoids. Con "fit_predict" se ejecuta
    # el algoritmo de K-Medoids sobre todo el conjunto ("X"), obteniendo k clusters:
    VI = None
    if nombre_metrica == "mahalanobis":
        VI = obtener_vi(X)
        matriz_distancias = pairwise_distances(X, metric = nombre_metrica, VI = VI)
        modelo_kmedoids = KMedoids(n_clusters = k, metric = "precomputed", random_state = semilla)
        clusters = modelo_kmedoids.fit_predict(matriz_distancias)
    else:
        modelo_kmedoids = KMedoids(n_clusters = k, metric = nombre_metrica, random_state = semilla)
        clusters = modelo_kmedoids.fit_predict(X)

    # Identificamos los positivos dentro de "y":
    mask_positivos = (y == 1)

    # Identificamos los clústeres que contengan algún ejemplo positivo:
    clusters_con_pos = np.unique(clusters[mask_positivos])

    # Se calcula la diferencia entre todos los clústeres y los anteriores, es decir,
    # los clústeres donde no hay ningún positivo. Si este es el caso, entonces
    # todas las muestras de ese clúster serán Negativos Fiables:
    clusters_sin_pos = np.setdiff1d(np.arange(k), clusters_con_pos)

    # Obtenemos los índices de los negativos fiables:
    indices_RN = np.where(np.isin(clusters, clusters_sin_pos))[0]

    print("\nIdentificacion de Negativos Fiables con CRNE completada:")
    print(f"\tPositivos: {len(X[y == 1])}")
    print(f"\tNo Etiquetados: {len(X[y == 0])}")
    print(f"\tNegativos Fiables: {len(indices_RN)}")

    return indices_RN