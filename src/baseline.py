"""Experiment group 1: baselines.

Run:  python -m src.baseline

* B0 guesses each emotion at random with its training frequency. It is the
  floor: any real model must beat it.
* B1-B5 are TF-IDF features + one logistic regression per emotion
  (one-vs-rest), the classic strong baseline for short-text classification.
  - B1 vs B2: does class weighting + threshold tuning help the rare emotions?
  - B2 vs B3: word vs character n-grams. Kinyarwanda is agglutinative, so one
    stem appears with many prefixes/suffixes (ababaye, yababaye, ababara);
    character n-grams share evidence across those forms, whole words cannot.
  - B4: both feature sets combined.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import EMOTIONS, MODELS_DIR, PROCESSED_DIR, SEED
from src.evaluate import compute_metrics, log_experiment, save_report, tune_thresholds


def word_tfidf():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w[\w']*\b|[^\w\s]")  # keeps ! ? and emojis


def char_tfidf():
    return TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True)


def ovr_logreg(balanced: bool):
    return OneVsRestClassifier(LogisticRegression(
        C=4, max_iter=2000, class_weight="balanced" if balanced else None, random_state=SEED))


# name -> (features, balanced class weights?, tune thresholds on dev?)
EXPERIMENTS = {
    "B1_word_tfidf_logreg_plain": (word_tfidf, False, False),
    "B2_word_tfidf_logreg_tuned": (word_tfidf, True, True),
    "B3_char_tfidf_logreg_tuned": (char_tfidf, True, True),
    "B4_word+char_tfidf_logreg_tuned": (
        lambda: FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())]), True, True),
}


def load():
    return [pd.read_csv(PROCESSED_DIR / f"{s}.csv") for s in ("train", "dev", "test")]


def random_baseline(train, dev, test):
    rng = np.random.default_rng(SEED)
    prior = train[EMOTIONS].mean().to_numpy()
    scores = []
    for df in (dev, test):
        pred = (rng.random((len(df), len(EMOTIONS))) < prior).astype(int)
        scores.append(compute_metrics(df[EMOTIONS], pred))
    log_experiment("B0_random_by_label_frequency", {"prior": prior.round(3).tolist()}, *scores)
    print(f"{'B0_random_by_label_frequency':34s} dev macro-F1 {scores[0]['macro_f1']:.3f} | "
          f"test macro-F1 {scores[1]['macro_f1']:.3f}")


def main():
    train, dev, test = load()
    random_baseline(train, dev, test)
    best = (None, -1.0)
    for name, (features, balanced, tune) in EXPERIMENTS.items():
        pipe = Pipeline([("tfidf", features()), ("clf", ovr_logreg(balanced))])
        pipe.fit(train["clean_text"], train[EMOTIONS])

        dev_probs = pipe.predict_proba(dev["clean_text"])
        thresholds = tune_thresholds(dev[EMOTIONS], dev_probs) if tune else np.full(len(EMOTIONS), 0.5)
        test_probs = pipe.predict_proba(test["clean_text"])
        dev_m = compute_metrics(dev[EMOTIONS], dev_probs >= thresholds)
        test_pred = (test_probs >= thresholds).astype(int)
        test_m = compute_metrics(test[EMOTIONS], test_pred)

        log_experiment(name, {"balanced": balanced, "thresholds": thresholds.round(2).tolist()},
                       dev_m, test_m)
        save_report(name, test, test_pred, test_probs)
        print(f"{name:34s} dev macro-F1 {dev_m['macro_f1']:.3f} | test macro-F1 "
              f"{test_m['macro_f1']:.3f}  micro-F1 {test_m['micro_f1']:.3f}")
        if dev_m["macro_f1"] > best[1]:
            best = (name, dev_m["macro_f1"])
            MODELS_DIR.mkdir(exist_ok=True)
            joblib.dump({"pipeline": pipe, "thresholds": thresholds}, MODELS_DIR / "baseline.joblib")
    print(f"Best on dev: {best[0]} -> models/baseline.joblib")


if __name__ == "__main__":
    main()
