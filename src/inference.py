"""The fine-tuned emotion classifier, as used by the web app and src/predict.py.

Kept free of training/evaluation dependencies (sklearn, matplotlib) so the web
app installs only what inference needs.
"""
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import EMOTIONS
from src.data import normalize

MODEL_ID = os.environ.get("MODEL_ID", "Alliane/kinyarwanda-emotion-afroxlmr")


class EmotionClassifier:
    """Text -> one sigmoid probability per emotion -> per-emotion tuned thresholds."""

    def __init__(self, model_id: str = MODEL_ID, dtype: str = "float32"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).eval()
        with open(hf_hub_download(model_id, "thresholds.txt")) as f:
            self.thresholds = np.array([float(x) for x in f.read().split()])

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer([normalize(t) for t in texts[i:i + batch_size]], truncation=True,
                                 max_length=128, padding=True, return_tensors="pt")
            out.append(torch.sigmoid(self.model(**enc).logits.float()).numpy())
        return np.concatenate(out)

    def predict(self, text: str) -> dict:
        probs = self.predict_proba([text])[0]
        return {e: {"probability": float(p), "threshold": float(t), "predicted": bool(p >= t)}
                for e, p, t in zip(EMOTIONS, probs, self.thresholds)}

    def tokens(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(normalize(text))
