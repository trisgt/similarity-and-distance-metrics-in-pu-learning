from config.paths_config import GENUINE_PU_DATASETS, SPLIT_DATASETS, RESULTADOS
from preprocesado.pr_breast_cancer import NUM_FOLDS, GUARDADO_NPY, CL_POSITIVAS, PORCENTAJE_POS
from entrenamiento_y_eval import entrenamiento_y_eval_train_test_separate

# Rutas para los folds, tanto PU como no PU. Para el conjunto de test,
# se usarán siempre los no PU. Para el de entrenamiento, depende:
RUTA_FOLDS = SPLIT_DATASETS / "breast_cancer"
RUTA_FOLDS_GEN_PU = GENUINE_PU_DATASETS / "breast_cancer"

# Nombre del dataset, utilizado para dar información:
NOMBRE = "Breast Cancer Wisconsin"

# Ruta del .txt para guardar los resultados:
RUTA_TXT = RESULTADOS / "breast_cancer.txt"

# Función principal:
if __name__ == "__main__":
    entrenamiento_y_eval_train_test_separate(
        RUTA_FOLDS,
        RUTA_FOLDS_GEN_PU,
        GUARDADO_NPY,
        NUM_FOLDS,
        CL_POSITIVAS,
        PORCENTAJE_POS,
        NOMBRE,
        RUTA_TXT        
    )