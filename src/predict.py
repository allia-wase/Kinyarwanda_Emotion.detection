"""Command-line inference and test-set re-scoring for the published model.

Predict:   python -m src.predict "ndumva mfite agahinda kenshi uyu munsi"
Evaluate:  python -m src.predict --evaluate [--dtype bfloat16]

--evaluate re-scores the whole test set, to check that the published model
reproduces the reported number, and that loading it in bfloat16 (half the
memory, needed to fit the free Streamlit Cloud tier) does not change results.
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.config import EMOTIONS, PROCESSED_DIR
from src.evaluate import compute_metrics
from src.inference import EmotionClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    clf = EmotionClassifier(dtype=args.dtype)
    if args.text:
        for e, r in clf.predict(args.text).items():
            print(f"{e:9s} p={r['probability']:.3f} (threshold {r['threshold']:.2f}) "
                  f"{'<- predicted' if r['predicted'] else ''}")
    if args.evaluate:
        test = pd.read_csv(PROCESSED_DIR / "test.csv")
        start = time.time()
        probs = clf.predict_proba(test["text"].tolist())
        m = compute_metrics(test[EMOTIONS], probs >= clf.thresholds)
        print(f"[{args.dtype}] test macro-F1 {m['macro_f1']:.4f}, micro-F1 {m['micro_f1']:.4f}, "
              f"{(time.time() - start) / len(test) * 1000:.0f} ms/text")
        np.save(PROCESSED_DIR.parent / f"test_probs_{args.dtype}.npy", probs)


if __name__ == "__main__":
    main()
