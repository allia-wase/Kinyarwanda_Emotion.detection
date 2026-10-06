"""Experiment group 2: BiLSTM classifiers.

Run:  python -m src.bilstm            (needs models/word2vec_kin.pt; ~10 min on CPU)

Architecture (one text in, six independent yes/no probabilities out):

  tokens -> Embedding (100-d) -> BiLSTM (128 per direction) -> pooling -> dropout
         -> Linear(256 -> 6) -> sigmoid per emotion

* The BiLSTM reads the sentence left-to-right and right-to-left, so each word's
  hidden state knows its context on both sides (negation such as "ntabwo ...",
  "not ...", can flip the meaning of a later word).
* Pooling turns one vector per word into one vector per text: either the
  element-wise max over words, or additive attention (Bahdanau et al., 2015)
  that learns a weight per word. The attention weights double as an explanation
  of which words drove a prediction, which the web app displays.
* Sigmoid + binary cross-entropy (not softmax) because a text can carry several
  emotions or none. pos_weight up-weights the rare positives, the neural
  equivalent of class_weight="balanced" in the baseline.

Experiments (each run with 3 random seeds; mean and std reported, because with
2.4k training texts a single run can be lucky or unlucky):
  L1  random embeddings,               max pooling
  L2  Word2Vec embeddings, frozen,     max pooling
  L3  Word2Vec embeddings, fine-tuned, max pooling
  L4  Word2Vec embeddings, fine-tuned, attention pooling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.config import EMOTIONS, MODELS_DIR, PROCESSED_DIR
from src.evaluate import compute_metrics, log_experiment, save_report, tune_thresholds
from src.text import Vocab, tokenize

MAX_LEN, EMB_DIM, HIDDEN, DROPOUT = 64, 100, 128, 0.4
BATCH, LR, MAX_EPOCHS, PATIENCE = 32, 2e-3, 30, 5
SEEDS = [42, 43, 44]

EXPERIMENTS = {
    "L1_bilstm_random_emb_maxpool": dict(w2v=False, freeze=False, pooling="max"),
    "L2_bilstm_w2v_frozen_maxpool": dict(w2v=True, freeze=True, pooling="max"),
    "L3_bilstm_w2v_finetuned_maxpool": dict(w2v=True, freeze=False, pooling="max"),
    "L4_bilstm_w2v_finetuned_attention": dict(w2v=True, freeze=False, pooling="attention"),
}


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, pooling="max", embeddings=None, freeze=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
        self.embedding.weight.requires_grad = not freeze
        self.lstm = nn.LSTM(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.pooling = pooling
        self.att_proj = nn.Linear(2 * HIDDEN, 2 * HIDDEN)
        self.att_vec = nn.Linear(2 * HIDDEN, 1, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.out = nn.Linear(2 * HIDDEN, len(EMOTIONS))

    def forward(self, ids, lengths, return_attention=False):
        x = self.dropout(self.embedding(ids))                              # (B, T, E)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True,
                                   total_length=ids.size(1))               # (B, T, 2H)
        mask = ids != 0
        if self.pooling == "max":
            pooled, weights = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values, None
        else:
            scores = self.att_vec(torch.tanh(self.att_proj(h))).squeeze(-1)  # (B, T)
            weights = torch.softmax(scores.masked_fill(~mask, -1e9), dim=1)
            pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
        logits = self.out(self.dropout(pooled))
        return (logits, weights) if return_attention else logits


def build_vocab(train_texts, use_w2v):
    """Random-embedding runs use the training vocabulary; Word2Vec runs use the
    Word2Vec vocabulary, so words never seen in the labelled data still get a
    meaningful vector. The rest of the vocabulary gets a small random vector."""
    if not use_w2v:
        return Vocab.build(train_texts, min_count=2), None
    w2v = torch.load(MODELS_DIR / "word2vec_kin.pt")
    extra = sorted({t for text in train_texts for t in tokenize(text)} - set(w2v["itos"]))
    vocab = Vocab(w2v["itos"][2:] + extra)
    emb = torch.randn(len(vocab), EMB_DIM) * w2v["vectors"].std()
    emb[: len(w2v["itos"])] = w2v["vectors"]
    emb[0] = 0
    return vocab, emb


def encode(df, vocab):
    seqs = [vocab.encode(t, MAX_LEN) or [1] for t in df["clean_text"]]
    lengths = torch.tensor([len(s) for s in seqs])
    ids = torch.zeros(len(seqs), MAX_LEN, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, lengths, torch.tensor(df[EMOTIONS].to_numpy(), dtype=torch.float)


@torch.no_grad()
def predict_proba(model, ids, lengths):
    model.eval()
    return torch.cat([torch.sigmoid(model(ids[i:i + 256], lengths[i:i + 256]))
                      for i in range(0, len(ids), 256)]).numpy()


def train_one(cfg, seed, data, vocab, emb, pos_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)
    (tr_ids, tr_len, tr_y), (dv_ids, dv_len, dv_y) = data
    model = BiLSTMClassifier(len(vocab), cfg["pooling"], emb, cfg["freeze"])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, best_state, bad = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for idx in torch.randperm(len(tr_ids)).split(BATCH):
            loss = loss_fn(model(tr_ids[idx], tr_len[idx]), tr_y[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        probs = predict_proba(model, dv_ids, dv_len)
        f1 = compute_metrics(dv_y.numpy(), probs >= tune_thresholds(dv_y.numpy(), probs))["macro_f1"]
        if f1 > best:
            best, bad, best_epoch = f1, 0, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:  # early stopping: dev score stopped improving
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def main():
    train, dev, test = (pd.read_csv(PROCESSED_DIR / f"{s}.csv") for s in ("train", "dev", "test"))
    pos = train[EMOTIONS].sum().to_numpy()
    pos_weight = torch.tensor(np.sqrt((len(train) - pos) / pos), dtype=torch.float)

    for name, cfg in EXPERIMENTS.items():
        vocab, emb = build_vocab(train["clean_text"], cfg["w2v"])
        tr, dv, te = encode(train, vocab), encode(dev, vocab), encode(test, vocab)
        oov = float((te[0][te[0] != 0] == 1).float().mean())
        runs = []
        for seed in SEEDS:
            model, epoch = train_one(cfg, seed, (tr, dv), vocab, emb, pos_weight)
            dev_p, test_p = predict_proba(model, *dv[:2]), predict_proba(model, *te[:2])
            thr = tune_thresholds(dev[EMOTIONS], dev_p)
            runs.append(dict(seed=seed, epoch=epoch, model=model, thr=thr, test_p=test_p,
                             dev=compute_metrics(dev[EMOTIONS], dev_p >= thr),
                             test=compute_metrics(test[EMOTIONS], test_p >= thr)))
            print(f"  {name} seed {seed}: best epoch {epoch}, dev {runs[-1]['dev']['macro_f1']:.3f}, "
                  f"test {runs[-1]['test']['macro_f1']:.3f}")

        mean = lambda split: {k: float(np.mean([r[split][k] for r in runs])) for k in runs[0][split]}
        test_std = float(np.std([r["test"]["macro_f1"] for r in runs]))
        log_experiment(name, cfg | {"seeds": SEEDS, "test_macro_f1_std": round(test_std, 4),
                                    "test_oov_rate": round(oov, 4), "vocab": len(vocab),
                                    "epochs": [r["epoch"] for r in runs]},
                       mean("dev"), mean("test"))
        rep = max(runs, key=lambda r: r["dev"]["macro_f1"])  # chosen on dev, never on test
        save_report(name, test, (rep["test_p"] >= rep["thr"]).astype(int), rep["test_p"])
        torch.save({"state_dict": rep["model"].state_dict(), "itos": vocab.itos, "config": cfg,
                    "thresholds": rep["thr"].tolist()}, MODELS_DIR / f"{name}.pt")
        print(f"{name}: test macro-F1 {mean('test')['macro_f1']:.3f} ± {test_std:.3f} "
              f"(OOV on test {oov:.1%})")


if __name__ == "__main__":
    main()
