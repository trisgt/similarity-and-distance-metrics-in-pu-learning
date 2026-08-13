from config.paths_config import ENGINEERED_DATASETS, SPLIT_DATASETS, RESULTADOS
from preprocesado.pr_bank_marketing import NUM_FOLDS, GUARDADO_NPY, CL_POSITIVAS, PORCENTAJE_POS
from entrenamiento_y_eval import entrenamiento_y_eval

# Rutas para los folds, tanto PU como no PU. Para el conjunto de test,
# se usarán siempre los no PU. Para el de entrenamiento, depende:
RUTA_FOLDS = SPLIT_DATASETS / "bank_marketing"
RUTA_FOLDS_PU = ENGINEERED_DATASETS / "bank_marketing"

# Nombre del dataset, utilizado para dar información:
NOMBRE = "Bank Marketing"

# Ruta del .txt para guardar los resultados:
RUTA_TXT = RESULTADOS / "bank_marketing.txt"

# Función principal:
if __name__ == "__main__":
    entrenamiento_y_eval(
        RUTA_FOLDS,
        RUTA_FOLDS_PU,
        GUARDADO_NPY,
        NUM_FOLDS,
        CL_POSITIVAS,
        PORCENTAJE_POS,
        NOMBRE,
        RUTA_TXT
    )