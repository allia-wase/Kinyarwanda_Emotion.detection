"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

HF_DATASET = "brighter-dataset/BRIGHTER-emotion-categories"
HF_CONFIG = "kin"
PARQUET_URL = ("https://huggingface.co/api/datasets/" + HF_DATASET +
               "/parquet/" + HF_CONFIG + "/{split}/0.parquet")

SEED = 42
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
