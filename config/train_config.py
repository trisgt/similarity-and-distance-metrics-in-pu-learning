# Selección de métodos para Negativos Fiables y Aprendizaje:
MET_NEG_FIABLES = "Rocchio"
MET_APRENDIZAJE = "Random Forest"

# Parámetros para Negativos Fiables (en Two-Step Methods):
METRICA = "euclidean"
K = 10                  # En KNN, K-Means, K-Medoids y CRNE
PORCENTAJE_RN = 0.25    # En KNN, K-Means y K-Medoids
SEMILLA_RN = 1          # En K-Means, K-Medoids y CRNE

# Parámetros para Aprendizaje:
PENALTY = "l2"          # Con Regresión Logística
C = 1.0                 # Con Regresión Logística
N_ESTIMATORS = 100      # Con Random Forest
CRITERION = "gini"      # Con Random Forest
MAX_DEPTH = None        # Con Random Forest
HIDDEN_LAYER_SIZES = (100, )    # Con Perceptrón Multicapa
ACTIVATION = "relu"     # Con Perceptrón Multicapa
SEMILLA_L = 1
BALANCEO_CLASES = True