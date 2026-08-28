# Parámetros comunes para el split del dataset en k-folds:
NUM_FOLDS = 5
SEMILLA_SPLIT = 42

# Parámetros comunes para la conversión del dataset a PU:
PORCENTAJE_POS = 0.5    # Porcentaje de positivos visibles tras la conversión a PU
SEMILLA_PR = 42

# Parámetros comunes para los embeddings en datasets de texto:
BATCH_SIZE = 32
DEVICE = ["cpu", "cpu", "cpu", "cpu"]