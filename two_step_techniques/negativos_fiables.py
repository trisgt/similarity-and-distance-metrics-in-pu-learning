import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from two_step_techniques.metricas_distancia import comprobar_nombre_metrica, obtener_metrica, obtener_vi


# Método de Rocchio. En cada método se utiliza un dataset ya convertido en PU (engineered):
def rocchio(X, y, nombre_metrica):
    # Transformamos "X" e "y" en arrays de numpy para una mayor eficiencia:
    X = np.asarray(X)
    y = np.asarray(y)

    # Comprobamos que la métrica sea válida:
    metrica = comprobar_nombre_metrica(nombre_metrica)

    # Separamos las muestras Positivas y las No Etiquetadas:
    P = X[y == 1]
    U = X[y == 0]

    # Cada clase se representa mediante un prototpio de Rocchio (centroide medio).
    # Se calculan el de los Positivos y el de los No Etiquetados:
    prototipo_p = np.mean(P, axis = 0).reshape(1, -1)
    prototipo_u = np.mean(U, axis = 0).reshape(1, -1)

    # Calculamos las distancias de todos los No Etiquetados a ambos prototipos:
    if metrica == "mahalanobis": # Para Mahalanobis se necesita VI
        VI = obtener_vi(X)
        dist_prot_p = pairwise_distances(U, prototipo_p, metrica, VI = VI).flatten()
        dist_prot_u = pairwise_distances(U, prototipo_u, metrica, VI = VI).flatten()
    else:
        dist_prot_p = pairwise_distances(U, prototipo_p, metrica).flatten()
        dist_prot_u = pairwise_distances(U, prototipo_u, metrica).flatten()

    # Escojemos los negativos fiables: estos serán los que estén más cerca del prototipo de
    # No Etiquetados que del prototipo de Positivos. Se comprueba para cada muestra:
    RN = U[dist_prot_u < dist_prot_p]

    print("\nIdentificacion de Negativos Fiables con Rocchio completada:")
    print(f"\tMetrica: {nombre_metrica}")
    print(f"\tPositivos: {len(P)}")
    print(f"\tNo Etiquetados: {len(U)}")
    print(f"\tNegativos Fiables: {len(RN)}")

    return RN


# KNN (K-Nearest Neighbours):
def knn(X, y, nombre_metrica, k, porcentaje_rn):
    # Transformamos "X" e "y" en arrays de numpy para una mayor eficiencia:
    X = np.asarray(X)
    y = np.asarray(y)

    # Comprobamos que la métrica sea válida:
    metrica = comprobar_nombre_metrica(nombre_metrica)

    # Separamos las muestras Positivas y las No Etiquetadas:
    P = X[y == 1]
    U = X[y == 0]

    # Creamos un modelo de KNN:
    if metrica == "mahalanobis":
        VI = obtener_vi(P)
        knn_model = NearestNeighbors(n_neighbors = k, metric = metrica, metric_params = {"VI": VI})
    else:
        knn_model = NearestNeighbors(n_neighbors = k, metric = metrica)

    # Con "fit" hacemos que cree un mapa de los Positivos.
    # El modelo de KNN calculará las distancias a esos puntos:
    knn_model.fit(P)

    # Para cada No Etiquetado, obtenemos su distancia media a los k vecinos positivos más cercanos:
    distancias, _ = knn_model.kneighbors(U) # Sólo nos interesan las distancias obtenidas, no los índices de los puntos
    dist_medias = np.mean(distancias, axis = 1)

    # Se deciden cuáles son los No Etiquetados más alejados de los positivos (Neg. fiables).
    # Se utiliza el porcentaje de negativos fiables para separarlos:
    threshold = np.quantile(dist_medias, 1 - porcentaje_rn)
    RN = U[dist_medias >= threshold]

    print("\nIdentificacion de Negativos Fiables con KNN completada:")
    print(f"\tMetrica: {nombre_metrica}; k = {k}; porcentaje RN: {porcentaje_rn}")
    print(f"\tPositivos: {len(P)}")
    print(f"\tNo Etiquetados: {len(U)}")
    print(f"\tNegativos Fiables: {len(RN)}")

    return RN


# K-Means:
def kmeans(X, y, nombre_metrica, k, porcentaje_rn, semilla = 1):
    # Transformamos "X" e "y" en arrays de numpy para una mayor eficiencia:
    X = np.asarray(X)
    y = np.asarray(y)

    # Comprobamos que la métrica sea válida y obtenemos una métrica de scipy:
    metrica = comprobar_nombre_metrica(nombre_metrica)
    metrica_sc = obtener_metrica(metrica)

    # Separamos las muestras Positivas y las No Etiquetadas:
    P = X[y == 1]
    U = X[y == 0]

    # Creamos un modelo de K-Means. Con "fit_predict" se ejecuta el algoritmo de K-Means
    # sobre los No Etiquetados, obteniendo k clusters que ya han convergido:
    kmeans_model = KMeans(n_clusters = k, random_state = semilla)
    clusters_u = kmeans_model.fit_predict(U)

    # Obtenemos los centroides de cada cluster de los No Etiquetados, y el centroide (prototipo) positivo:
    centroides_u = kmeans_model.cluster_centers_
    prototipo_p = np.mean(P, axis = 0)

    # En caso de distancia de Mahalanobis:
    VI = None
    if nombre_metrica == "mahalanobis":
        VI = obtener_vi(P)

    # Calculamos la distancia de cada centroide (cluster) negativo al positivo:
    dist_centroides = []
    for c in centroides_u:
        if nombre_metrica == "mahalanobis":
            distancia = metrica_sc(c, prototipo_p, VI = VI)
        else:
            distancia = metrica_sc(c, prototipo_p)
        dist_centroides.append(distancia)
    dist_centroides = np.array(dist_centroides)

    # Ordenamos los clusters por distancia, de más lejanos a más cercanos al positivo:
    dist_centroides_ord = np.argsort(dist_centroides)[::-1]

    # Seleccionamos la mitad de estos clusters como candidatos a negativos:
    num_clusters_neg = max(1, k // 2)
    clusters_neg = dist_centroides_ord[:num_clusters_neg]

    # Seleccionamos las muestras individuales candidatas dentro de estos clusters:
    candidatos = U[np.isin(clusters_u, clusters_neg)]

    # Si no hay candidatos, devolver vacio...?

    # Se calculan las distancias de los candidatos al prototipo positivo:
    dist_candidatos = []
    for m in candidatos:
        if nombre_metrica == "mahalanobis":
            distancia = metrica_sc(m, prototipo_p, VI = VI)
        else:
            distancia = metrica_sc(m, prototipo_p)
        dist_candidatos.append(distancia)
    dist_candidatos = np.array(dist_candidatos)

    # Ordenamos los candidatos por distancia, de más lejanos a más cercanos al positivo:
    dist_candidatos_ord = np.argsort(dist_candidatos)[::-1]

    # Seleccionamos un porcentaje de los candidatos como negativos fiables:
    num_rn = int(len(U) * porcentaje_rn)
    RN = candidatos[dist_candidatos_ord[:num_rn]]

    print("\nIdentificacion de Negativos Fiables con K-Means completada:")
    print(f"\tMetrica: {nombre_metrica}; k = {k}; porcentaje RN: {porcentaje_rn}; semilla = {semilla}")
    print(f"\tPositivos: {len(P)}")
    print(f"\tNo Etiquetados: {len(U)}")
    print(f"\tNegativos Fiables: {len(RN)}")

    return RN


# K-Medoids:
def kmedoids(X, y, nombre_metrica, k, porcentaje_rn, semilla = 1):
    # Transformamos "X" e "y" en arrays de numpy para una mayor eficiencia:
    X = np.asarray(X)
    y = np.asarray(y)

    # Comprobamos que la métrica sea válida. Además, obtenemos una métrica de scipy:
    metrica = comprobar_nombre_metrica(nombre_metrica)
    metrica_sc = obtener_metrica(metrica)

    # Separamos las muestras Positivas y las No Etiquetadas:
    P = X[y == 1]
    U = X[y == 0]

    # Creamos un modelo de K-Medoids. Con "fit_predict" se ejecuta el algoritmo de K-Medoids
    # sobre los No Etiquetados, obteniendo k clusters que ya han convergido:
    kmedoids_model = KMedoids(n_clusters = k, metric = metrica, random_state = semilla)
    clusters_u = kmedoids_model.fit_predict(U)

    # Obtenemos los centroides (medoides) de cada cluster de los No Etiquetados. La diferencia con K-Means
    # es que estos son puntos reales del dataset. Además, obtenemos el centroide (prototipo) positivo:
    medoides_u = kmedoids_model.medoid_indices_
    prototipo_p = np.mean(P, axis = 0)

    # En caso de distancia de Mahalanobis:
    VI = None
    if nombre_metrica == "mahalanobis":
        VI = obtener_vi(P)

    # Calculamos la distancia de cada medoide (cluster) negativo al positivo:
    dist_medoides = []
    for idx in medoides_u:
        m = U[idx]
        if nombre_metrica == "mahalanobis":
            distancia = metrica_sc(m, prototipo_p, VI = VI)
        else:
            distancia = metrica_sc(m, prototipo_p)
        dist_medoides.append(distancia)
    dist_medoides = np.array(dist_medoides)

    # Ordenamos los clusters por distancia, de más lejanos a más cercanos al positivo:
    dist_medoides_ord = np.argsort(dist_medoides)[::-1]

    # Seleccionamos la mitad de estos clusters como candidatos a negativos:
    num_clusters_neg = max(1, k // 2)
    clusters_neg = dist_medoides_ord[:num_clusters_neg]

    # Seleccionamos las muestras individuales candidatas dentro de estos clusters:
    candidatos = U[np.isin(clusters_u, clusters_neg)]

    # Si no hay candidatos, devolver vacio...?

    # Se calculan las distancias de los candidatos al prototipo positivo:
    dist_candidatos = []
    for m in candidatos:
        if nombre_metrica == "mahalanobis":
            distancia = metrica_sc(m, prototipo_p, VI = VI)
        else:
            distancia = metrica_sc(m, prototipo_p)
        dist_candidatos.append(distancia)
    dist_candidatos = np.array(dist_candidatos)

    # Ordenamos los candidatos por distancia, de más lejanos a más cercanos al positivo:
    dist_candidatos_ord = np.argsort(dist_candidatos)[::-1]

    # Seleccionamos un porcentaje de los candidatos como negativos fiables:
    num_rn = int(len(U) * porcentaje_rn)
    RN = candidatos[dist_candidatos_ord[:num_rn]]

    print("\nIdentificacion de Negativos Fiables con K-Medoids completada:")
    print(f"\tMetrica: {nombre_metrica}; k = {k}; porcentaje RN: {porcentaje_rn}; semilla = {semilla}")
    print(f"\tPositivos: {len(P)}")
    print(f"\tNo Etiquetados: {len(U)}")
    print(f"\tNegativos Fiables: {len(RN)}")

    return RN


# CRNE:
def crne(X, y, nombre_metrica, k, semilla = 1):
    # Transformamos "X" e "y" en arrays de numpy para una mayor eficiencia:
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Comprobamos que la métrica sea válida y obtenemos una métrica de scipy:
    metrica = comprobar_nombre_metrica(nombre_metrica)
    metrica_sc = obtener_metrica(metrica)

    # Hacemos clustering con K-Means, y utilizamos "fit_predict" en todo el conjunto:
    kmeans_model = KMeans(n_clusters = k, random_state = semilla)
    clusters = kmeans_model.fit_predict(X)

    # Detectamos los positivos en y:
    p_mask = (y == 1)

    # Identificamos clusters que no tengan positivos:
    clusters_rn = []
    for c in range(k):
        cluster_idx = np.where(clusters == c)[0]

        # Si no hay ningun positivo en el cluster, se convierte en un cluster RN:
        if not np.any(p_mask[cluster_idx]):
            clusters_rn.append(c)
    
    # Extraemos los RN de cada cluster:
    RN = X[np.isin(clusters, clusters_rn)]

    print("\nIdentificacion de Negativos Fiables con CRNE completada:")
    print(f"\tMetrica: {nombre_metrica}; k = {k}; semilla = {semilla}")
    print(f"\tPositivos: {len(X[y == 1])}")
    print(f"\tNo Etiquetados: {len(X[y == 0])}")
    print(f"\tNegativos Fiables: {len(RN)}")

    return RN