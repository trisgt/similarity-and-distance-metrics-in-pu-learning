from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATOS = ROOT / "datos"
RESULTADOS = ROOT / "resultados"

DATASETS = DATOS / "datasets"
ENGINEERED_DATASETS = DATOS / "engineered_datasets"
SPLIT_DATASETS = DATOS / "split_datasets"