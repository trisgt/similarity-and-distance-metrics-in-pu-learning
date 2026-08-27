from pathlib import Path

# Raíz (como un Path):
# Raíz, guardada como un Path:
ROOT = Path(__file__).resolve().parent.parent

# Rutas para datos y resultados:
DATOS = ROOT / "datos"
RESULTADOS = ROOT / "resultados"

# Rutas para los datasets, dentro de datos:
DATASETS = DATOS / "datasets"
ENGINEERED_DATASETS = DATOS / "engineered_datasets"
SPLIT_DATASETS = DATOS / "split_datasets"
GENUINE_PU_DATASETS = DATOS / "genuine_pu_datasets"