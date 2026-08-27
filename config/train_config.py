# Selección de métodos para la obtención de Negativos Fiables y de Aprendizaje:
MET_NEG_FIABLES = "KNN"
MET_APRENDIZAJE = "Logistic Regression"

# Parámetros comunes para la obtención de Negativos Fiables (dentro de los Two-Step Methods):
METRICA = "euclidean"
K = 10                  # En KNN, K-Means, K-Medoids y CRNE
PORCENTAJE_RN = 0.2     # En KNN, K-Means y K-Medoids
SEMILLA_RN = 42         # En K-Means, K-Medoids y CRNE

# Parámetros comunes para el Aprendizaje:
BALANCEO_CLASES = True
SEMILLA_L = 42