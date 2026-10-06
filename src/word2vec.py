"""Kinyarwanda Word2Vec (skip-gram with negative sampling), implemented in PyTorch.

Run:  python -m src.word2vec          (about 5-10 minutes on a laptop CPU)
Writes models/word2vec_kin.pt and results/word2vec_neighbours.txt.

Why: the labelled emotion data is only 2,451 short texts, far too little to learn
what Kinyarwanda words mean. Word2Vec learns word vectors from *unlabelled* text
by predicting which words appear near each other (Mikolov et al., 2013), so
words used in similar contexts (ubwoba "fear", impungenge "worry") end up with
similar vectors. Those vectors then initialise the BiLSTM's embedding layer.

Corpus: the 11,089 unique KINNEWS news articles (Niyongabo et al., 2020), about
3.9M words. Any article containing a BRIGHTER dev/test sentence is dropped
first so no evaluation text is ever seen, even without its label.

The algorithm, step by step:
 1. Subsample very frequent words (ni, ku, mu...): they appear next to
    everything and teach little (Mikolov et al., 2013, eq. 5).
 2. For every word (the "centre"), the words within a random window of 1-5
    positions are its "context" -> positive (centre, context) pairs.
 3. For every positive pair draw K random "negative" words from the unigram
    distribution raised to 0.75.
 4. Two embedding tables: one for centre words (W_in), one for context words
    (W_out). Loss = -log sigmoid(in . out_pos) - sum log sigmoid(-in . out_neg).
    This pushes true neighbours together and random words apart.
 5. W_in is kept as the final word vectors.
"""
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ROOT, SEED
from src.data import normalize
from src.text import Vocab, tokenize

EXTERNAL = ROOT / "data" / "external"
KINNEWS_URL = "https://github.com/saradhix/kinnews_kirnews/raw/master/KINNEWS.zip"

DIM, WINDOW, NEGATIVES, MIN_COUNT = 100, 5, 5, 5
EPOCHS, BATCH, LR, SUBSAMPLE_T = 3, 4096, 0.003, 1e-4
PROBE_WORDS = ["ubwoba", "agahinda", "ibyishimo", "umujinya", "umwanda", "birababaje",
               "nziza", "urupfu", "ikibazo", "umukino"]


def load_corpus() -> list[str]:
    raw = EXTERNAL / "KINNEWS" / "raw"
    if not (raw / "train.csv").exists():
        import urllib.request
        EXTERNAL.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(KINNEWS_URL, EXTERNAL / "KINNEWS.zip")
        zipfile.ZipFile(EXTERNAL / "KINNEWS.zip").extractall(EXTERNAL)
    articles = pd.concat([pd.read_csv(raw / f"{s}.csv") for s in ("train", "test")])
    articles = articles.drop_duplicates("url")
    docs = (articles["title"].fillna("") + " . " + articles["content"].fillna("")).map(normalize)

    held_out = pd.concat([pd.read_csv(PROCESSED_DIR / f"{s}.csv") for s in ("dev", "test")])
    probes = [t.lower().strip(" .") for t in held_out["clean_text"] if len(t.split()) >= 8]
    lowered = docs.str.lower()
    leak = lowered.map(lambda d: any(p in d for p in probes))
    print(f"KINNEWS: {len(docs)} articles, dropped {int(leak.sum())} containing dev/test text")
    train_texts = pd.read_csv(PROCESSED_DIR / "train.csv")["clean_text"]  # tweets: adds social-media words
    return list(docs[~leak]) + list(train_texts)


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.w_in = nn.Embedding(vocab_size, dim, sparse=True)
        self.w_out = nn.Embedding(vocab_size, dim, sparse=True)
        nn.init.uniform_(self.w_in.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.w_out.weight)

    def forward(self, centre, context, negatives):
        v = self.w_in(centre)                                    # (B, D)
        pos = (v * self.w_out(context)).sum(-1)                  # (B,)
        neg = torch.bmm(self.w_out(negatives), v.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()


def make_pairs(doc_ids: list[np.ndarray], keep_prob: np.ndarray, rng) -> np.ndarray:
    """All (centre, context) pairs for one epoch, with subsampling and a random window."""
    pairs = []
    for ids in doc_ids:
        ids = ids[rng.random(len(ids)) < keep_prob[ids]]
        if len(ids) < 2:
            continue
        span = rng.integers(1, WINDOW + 1, size=len(ids))        # each centre's window size
        for off in range(1, WINDOW + 1):
            ok = span[:-off] >= off
            pairs.append(np.stack([ids[:-off][ok], ids[off:][ok]], 1))   # right context
            ok = span[off:] >= off
            pairs.append(np.stack([ids[off:][ok], ids[:-off][ok]], 1))   # left context
    pairs = np.concatenate(pairs)
    return pairs[rng.permutation(len(pairs))]


def nearest(vectors: torch.Tensor, vocab: Vocab, word: str, k: int = 8) -> list[str]:
    if word not in vocab.stoi:
        return ["(not in vocabulary)"]
    unit = F.normalize(vectors, dim=1)
    sims = unit @ unit[vocab.stoi[word]]
    return [vocab.itos[i] for i in sims.topk(k + 1).indices.tolist()[1:]]


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    docs = load_corpus()
    vocab = Vocab.build(docs, min_count=MIN_COUNT)
    doc_ids = [np.array(vocab.encode(d), dtype=np.int64) for d in docs]

    counts = np.bincount(np.concatenate(doc_ids), minlength=len(vocab)).astype(np.float64)
    counts[:2] = 0                                                 # never train on <pad>/<unk>
    freq = counts / counts.sum()
    keep_prob = np.minimum(1.0, np.sqrt(SUBSAMPLE_T / np.maximum(freq, 1e-12)) +
                           SUBSAMPLE_T / np.maximum(freq, 1e-12))
    neg_dist = torch.tensor(counts ** 0.75 / (counts ** 0.75).sum(), dtype=torch.float)
    print(f"vocab {len(vocab)} words (min count {MIN_COUNT}), corpus {int(counts.sum())} tokens")

    model = SkipGram(len(vocab), DIM)
    opt = torch.optim.SparseAdam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        pairs = torch.from_numpy(make_pairs(doc_ids, keep_prob, rng))
        total = 0.0
        for start in range(0, len(pairs), BATCH):
            batch = pairs[start:start + BATCH]
            negs = torch.multinomial(neg_dist, len(batch) * NEGATIVES, replacement=True)
            loss = model(batch[:, 0], batch[:, 1], negs.view(len(batch), NEGATIVES))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        print(f"epoch {epoch + 1}/{EPOCHS}: {len(pairs):,} pairs, mean loss {total / len(pairs):.4f}")

    vectors = model.w_in.weight.detach()
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"itos": vocab.itos, "vectors": vectors}, MODELS_DIR / "word2vec_kin.pt")
    lines = [f"{w:12s} -> {', '.join(nearest(vectors, vocab, w))}" for w in PROBE_WORDS]
    (RESULTS_DIR / "word2vec_neighbours.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
