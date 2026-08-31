# Selección de métodos para la obtención de Negativos Fiables y de Aprendizaje:
MET_NEG_FIABLES = "Rocchio"
MET_APRENDIZAJE = "MLP"

# Parámetros comunes para la obtención de Negativos Fiables (dentro de los Two-Step Methods):
METRICA = "jaccard"
K = 10                  # En KNN, K-Means, K-Medoids y CRNE
PORCENTAJE_RN = 0.7     # En KNN, K-Means y K-Medoids
SEMILLA_RN = 42         # En K-Means, K-Medoids y CRNE

# Parámetros comunes para el Aprendizaje:
BALANCEO_CLASES = True
SEMILLA_L = 42

# Opción para el entrenamiento y evaluación exclusivos de PU con Two-Step methods.
# Útil para ahorrar tiempo si entre experimentos sólo se cambian datos que afecten a este escenario:
SOLO_PU_CON_TS = False