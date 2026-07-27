from config.paths_config import *
from preprocesado.pr_bank_marketing import NUM_FOLDS, CL_POSITIVAS, PORCENTAJE_POS
from entrenamiento_y_eval import entrenamiento_y_eval

# Rutas para el dataset de entrenamiento (PU) y de test (no PU), en folds:
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "bank_marketing"
RUTA_FOLDS = SPLIT_DATASETS / "bank_marketing"

# Ruta y nombre del .txt para los resultados:
RUTA_TXT = RESULTADOS / "bank_marketing.txt"
NOMBRE = "Bank Marketing"

# Función principal:
if __name__ == "__main__":
    entrenamiento_y_eval(
        RUTA_FOLDS,
        RUTA_FOLDS_PU,
        NUM_FOLDS,
        CL_POSITIVAS,
        PORCENTAJE_POS,
        "BM", RUTA_TXT,
        "Bank Marketing")