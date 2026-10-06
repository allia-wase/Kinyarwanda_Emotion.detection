"""Download, clean and deduplicate the Kinyarwanda part of BRIGHTER.

Run:  python -m src.data
Writes data/processed/{train,dev,test}.csv and results/data_report.json.

Decisions (explained in the README, "Dataset"):
* The Hugging Face release lists every dev and test text twice, once under a
  Track A id and once under a Track C id, with identical labels. Scoring both
  copies would count each test text twice, so copies are dropped.
* The official train/dev/test split is kept, so results are comparable with
  SemEval-2025 Task 11 (Muhammad et al., 2025). Dev is used for model selection
  and threshold tuning; test is touched only for the final numbers.
"""
import json
import re
import unicodedata
import urllib.request

import pandas as pd

from src.config import EMOTIONS, PARQUET_URL, PROCESSED_DIR, RAW_DIR, RESULTS_DIR

SPLITS = ["train", "dev", "test"]
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    """Light normalisation, shared by training and the web app.

    * Kinyarwanda marks elided vowels with an apostrophe (n'abandi, y'imyaka) and
      the corpus mixes the typographic ’ with ASCII ', so both are unified.
    * Anonymised mentions/links become one token each, so a pasted real handle
      or URL in the web app looks the same as in training.
    * '#' is dropped from hashtags; the word itself is kept.
    * Emojis and punctuation (!!!, ?) are kept on purpose: they carry emotion.
    """
    text = unicodedata.normalize("NFKC", str(text)).translate(APOSTROPHES)
    text = re.sub(r"@<username>|@\w+", " @user ", text)
    text = re.sub(r"##url##|https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def download_if_missing() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = RAW_DIR / f"{split}.parquet"
        if not path.exists():
            print(f"Downloading {split} split")
            urllib.request.urlretrieve(PARQUET_URL.format(split=split), path)


def load_split(split: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(RAW_DIR / f"{split}.parquet")
    stats = {"rows_raw": len(df)}
    conflicting = df.groupby("text")[EMOTIONS].nunique().max(axis=1).loc[lambda s: s > 1]
    stats["conflicting_label_texts"] = len(conflicting)
    df = df[~df["text"].isin(conflicting.index)].drop_duplicates("text")
    stats["duplicates_removed"] = stats["rows_raw"] - len(df) - len(conflicting)
    df = df.assign(clean_text=df["text"].map(normalize))
    df = df[df["clean_text"].str.len() > 0]
    stats["rows"] = len(df)
    return df[["id", "text", "clean_text", *EMOTIONS]], stats


def main() -> None:
    download_if_missing()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report, texts = {}, {}
    for split in SPLITS:
        df, stats = load_split(split)
        df.to_csv(PROCESSED_DIR / f"{split}.csv", index=False)
        n_labels = df[EMOTIONS].sum(axis=1)
        words = df["clean_text"].str.split().str.len()
        stats |= {
            "emotion_counts": df[EMOTIONS].sum().astype(int).to_dict(),
            "labels_per_text": n_labels.value_counts().sort_index().astype(int).to_dict(),
            "words_mean": round(float(words.mean()), 1),
            "words_p95": float(words.quantile(0.95)),
            "with_mention": round(float(df["clean_text"].str.contains("@user").mean()), 3),
            "with_url": round(float(df["clean_text"].str.contains("URL").mean()), 3),
        }
        report[split], texts[split] = stats, set(df["text"])
        print(f"{split}: {stats['rows_raw']} rows -> {stats['rows']} unique texts")

    report["overlap"] = {"train_dev": len(texts["train"] & texts["dev"]),
                         "train_test": len(texts["train"] & texts["test"])}
    (RESULTS_DIR / "data_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["overlap"]))


if __name__ == "__main__":
    main()
