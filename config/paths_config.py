from pathlib import Path

# Raíz (como un Path):
ROOT = Path(__file__).resolve().parent

# Rutas para datos y resultados:
DATOS = ROOT / "datos"
RESULTADOS = ROOT / "resultados"

# Rutas para datasets, dentro de datos:
DATASETS = DATOS / "datasets"
ENGINEERED_DATASETS = DATOS / "engineered_datasets"
SPLIT_DATASETS = DATOS / "split_datasets"