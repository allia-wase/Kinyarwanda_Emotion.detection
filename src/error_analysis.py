"""Error analysis across experiments.

Run:  python -m src.error_analysis [EXPERIMENT ...]
      (default: the best experiment of each group present in results/)

Writes to results/error_analysis/:
  per_emotion.csv     precision / recall / F1 per emotion for each experiment
  confusions.csv      for single-emotion texts: gold emotion -> what was predicted instead
  slices.csv          macro-F1 on slices of the test set (tweets vs news-style, length, emoji, ...)
  hard_cases.csv      texts that EVERY analysed model gets wrong: candidates for label
                      noise, sarcasm, missing context. Reviewed by hand (native speaker).
  figures: per-emotion F1 bars, slice comparison.

Questions it answers:
  * Which emotions are hard, and is it precision (false alarms) or recall (misses)?
  * Which emotions get mistaken for each other?
  * Does the model do worse on tweets than on news-style sentences, on short
    texts, on texts with English code-switching?
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.config import EMOTIONS, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR

OUT = RESULTS_DIR / "error_analysis"
DEFAULT = ["B4_word+char_tfidf_logreg_tuned", "L3_bilstm_w2v_finetuned_maxpool",
           "T1_xlmr_base", "T3_afroxlmr_base"]
# Common English words seen in Kinyarwanda social media (code-switching signal).
ENGLISH = {"the", "and", "you", "is", "my", "love", "so", "i", "me", "for", "of", "to", "in",
           "it", "this", "that", "be", "are", "god", "bless", "thank", "thanks", "please",
           "happy", "sorry", "wow", "omg", "lol", "boyfriend", "sister", "baby", "nice", "good"}


def load_predictions(name: str, test: pd.DataFrame) -> np.ndarray:
    """Rebuild the full prediction matrix: errors.csv holds only wrong rows."""
    errors = pd.read_csv(RESULTS_DIR / name / "errors.csv").set_index("id")["pred"]
    gold = test[EMOTIONS].to_numpy().copy()
    for i, row_id in enumerate(test["id"]):
        if row_id in errors.index:
            pred = errors[row_id]
            gold[i] = [int(e in pred.split("+")) for e in EMOTIONS]
    return gold


def slices(test: pd.DataFrame) -> dict[str, pd.Series]:
    text = test["text"]
    words = test["clean_text"].str.split().str.len()
    tokens = test["clean_text"].str.lower().str.findall(r"[a-z']+")
    return {
        "tweet (has @mention)": text.str.contains("@<username>"),
        "no @mention (news-style)": ~text.str.contains("@<username>"),
        "has emoji": text.str.contains("[\U0001F300-\U0001FAFF☀-➿]"),
        "short (<=10 words)": words <= 10,
        "long (>25 words)": words > 25,
        "English code-switching": tokens.map(lambda t: len(ENGLISH & set(t)) >= 2),
        "no emotion (gold)": test[EMOTIONS].sum(axis=1) == 0,
        "2+ emotions (gold)": test[EMOTIONS].sum(axis=1) >= 2,
    }


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    gold = test[EMOTIONS].to_numpy()
    names = [n for n in names if (RESULTS_DIR / n / "errors.csv").exists()]
    preds = {n: load_predictions(n, test) for n in names}
    print("analysing:", ", ".join(names))

    # 1. Per-emotion precision / recall / F1
    rows = []
    for n, p in preds.items():
        pr, rc, f1, sup = precision_recall_fscore_support(gold, p, zero_division=0)
        for e, a, b, c, s in zip(EMOTIONS, pr, rc, f1, sup):
            rows.append({"experiment": n, "emotion": e, "precision": a, "recall": b, "f1": c,
                         "support": s, "predicted_count": int(p[:, EMOTIONS.index(e)].sum())})
    per_emotion = pd.DataFrame(rows).round(3)
    per_emotion.to_csv(OUT / "per_emotion.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(preds)
    for k, n in enumerate(preds):
        sub = per_emotion[per_emotion["experiment"] == n]
        ax.bar(np.arange(len(EMOTIONS)) + k * width, sub["f1"], width, label=n)
    ax.set_xticks(np.arange(len(EMOTIONS)) + width * (len(preds) - 1) / 2, EMOTIONS)
    ax.set_ylabel("test F1"); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    ax.set_title("Per-emotion F1 on the test set")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "per_emotion_f1.png", dpi=150); plt.close(fig)

    # 2. Cross-emotion confusions on texts with exactly one gold emotion
    single = gold.sum(axis=1) == 1
    rows = []
    for n, p in preds.items():
        for i in np.where(single)[0]:
            g = EMOTIONS[gold[i].argmax()]
            if p[i, gold[i].argmax()]:
                continue  # correct emotion found (maybe plus extras)
            predicted = [e for e, v in zip(EMOTIONS, p[i]) if v] or ["none"]
            for e in predicted:
                rows.append({"experiment": n, "gold": g, "predicted_instead": e})
    conf = (pd.DataFrame(rows).groupby(["experiment", "gold", "predicted_instead"]).size()
            .rename("count").reset_index().sort_values(["experiment", "count"], ascending=[True, False]))
    conf.to_csv(OUT / "confusions.csv", index=False)

    # 3. Slices
    rows = []
    for s_name, mask in slices(test).items():
        mask = mask.to_numpy()
        row = {"slice": s_name, "n_texts": int(mask.sum())}
        for n, p in preds.items():
            if s_name.startswith("no emotion"):  # F1 undefined: report share predicted as "none"
                row[n] = round(float((p[mask].sum(axis=1) == 0).mean()), 3)
            else:
                row[n] = round(f1_score(gold[mask], p[mask], average="macro", zero_division=0), 3)
        rows.append(row)
    sl = pd.DataFrame(rows)
    sl.to_csv(OUT / "slices.csv", index=False)

    # 4. Hard cases: wrong for every analysed model
    wrong_all = np.all([(p != gold).any(axis=1) for p in preds.values()], axis=0)
    names_of = lambda m: ["+".join(e for e, v in zip(EMOTIONS, r) if v) or "none" for r in m]
    hard = test.loc[wrong_all, ["id", "text"]].assign(gold=np.array(names_of(gold))[wrong_all])
    for n, p in preds.items():
        hard[f"pred_{n}"] = np.array(names_of(p))[wrong_all]
    hard["native_speaker_note"] = ""
    hard.to_csv(OUT / "hard_cases.csv", index=False)

    summary = {"experiments": names, "hard_cases_wrong_for_all": int(wrong_all.sum()),
               "test_size": len(test)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(per_emotion.pivot(index="emotion", columns="experiment", values="f1"))
    print(conf.groupby("experiment").head(6).to_string(index=False))
    print(sl.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
