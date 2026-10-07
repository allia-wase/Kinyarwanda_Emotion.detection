"""Experiment group 3: fine-tuning pretrained Transformers (run on a GPU, e.g. Colab T4).

Run all:   python -m src.transformer
Run one:   python -m src.transformer --only T3_afroxlmr_base --seeds 42

Research question: does a model that saw Kinyarwanda during pretraining detect
emotions better than one that did not?

  T1  XLM-R base        (Conneau et al., 2020)  100 languages, Kinyarwanda NOT among them
  T2  AfriBERTa large   (Ogueji et al., 2021)   trained from scratch on 11 African languages incl. Kinyarwanda
  T3  AfroXLMR base     (Alabi et al., 2022)    XLM-R base + continued pretraining on 17 African languages incl. Kinyarwanda
  T4  AfroXLMR large                            as T3, 2x deeper/wider: does scale help with 2.4k examples?
  T5  AfroXLMR base + LoRA (Hu et al., 2022)    only ~0.4% of weights trained: does parameter-efficient tuning keep up?

T1 vs T3 is the controlled comparison: identical architecture and tokenizer,
the only difference is the extra African-language pretraining.

How fine-tuning works here: the text is split into sub-word pieces by the
model's own SentencePiece tokenizer, passed through the pretrained Transformer
encoder, and the vector of the first token (<s>, which attends to every
other token) goes to a new, randomly initialised classification head
(dense layer -> tanh -> linear layer with 6 outputs). All weights (or, for
LoRA, small low-rank adapters + the classifier) are updated with AdamW, a
linear warm-up/decay learning-rate schedule, and binary cross-entropy with
pos_weight for the rare emotions (same loss as the BiLSTM).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import EMOTIONS, MODELS_DIR, PROCESSED_DIR
from src.evaluate import compute_metrics, log_experiment, save_report, tune_thresholds

EXPERIMENTS = {
    "T1_xlmr_base": dict(model="FacebookAI/xlm-roberta-base", lr=3e-5),
    "T2_afriberta_large": dict(model="castorini/afriberta_large", lr=3e-5),
    "T3_afroxlmr_base": dict(model="Davlan/afro-xlmr-base", lr=3e-5),
    "T4_afroxlmr_large": dict(model="Davlan/afro-xlmr-large", lr=1e-5),
    "T5_afroxlmr_base_lora": dict(model="Davlan/afro-xlmr-base", lr=5e-4, lora=True),
}
MAX_LEN, BATCH, EPOCHS, WARMUP, WEIGHT_DECAY, PATIENCE = 128, 16, 10, 0.1, 0.01, 3
SEEDS = [42, 43, 44]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cfg):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.model_max_length = MAX_LEN  # AfriBERTa ships without a max length
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=len(EMOTIONS), problem_type="multi_label_classification",
        id2label=dict(enumerate(EMOTIONS)), label2id={e: i for i, e in enumerate(EMOTIONS)})
    # AfroXLMR-large is stored in fp16 and recent transformers keep that dtype; mixed
    # precision needs fp32 master weights (fp16 is used only inside autocast).
    model = model.float()
    if cfg.get("lora"):
        from peft import LoraConfig, get_peft_model
        try:  # Colab ships torchao 0.10; peft >= 0.18 raises on it even though LoRA never uses it
            import peft.tuners.lora.torchao as peft_torchao
            peft_torchao.is_torchao_available = lambda: False
        except ImportError:
            pass
        model = get_peft_model(model, LoraConfig(
            task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query", "value"]))  # classifier head stays fully trainable
    return tok, model.to(DEVICE)


def make_loader(df, tok, shuffle):
    enc = tok(list(df["clean_text"]), truncation=True, max_length=MAX_LEN)
    labels = df[EMOTIONS].to_numpy(dtype=np.float32)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
              "labels": labels[i]} for i in range(len(df))]

    def collate(batch):
        out = tok.pad([{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch],
                      return_tensors="pt")
        out["labels"] = torch.tensor(np.stack([b["labels"] for b in batch]))
        return out
    return DataLoader(items, batch_size=BATCH, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            probs.append(torch.sigmoid(model(**batch).logits.float()).cpu())
    return torch.cat(probs).numpy()


def train_one(cfg, seed, train, dev, pos_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok, model = load_model(cfg)
    tr_loader, dv_loader = make_loader(train, tok, True), make_loader(dev, tok, False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    total = EPOCHS * len(tr_loader)
    warm = int(WARMUP * total)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warm-up, then linear decay to 0
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    n_trainable = sum(p.numel() for p in params)

    best, best_state, bad, best_epoch = -1.0, None, 0, 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in tr_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):  # fp16 on GPU: ~2x faster
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        probs = predict_proba(model, dv_loader)
        f1 = compute_metrics(dev[EMOTIONS], probs >= tune_thresholds(dev[EMOTIONS], probs))["macro_f1"]
        print(f"    epoch {epoch + 1}: last batch loss {loss.item():.4f}, dev macro-F1 {f1:.4f}")
        if f1 > best:
            best, bad, best_epoch = f1, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return tok, model, best_epoch, n_trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="experiment names to run (default: all)")
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()
    print(f"device: {DEVICE}")

    train, dev, test = (pd.read_csv(PROCESSED_DIR / f"{s}.csv") for s in ("train", "dev", "test"))
    pos = train[EMOTIONS].sum().to_numpy()
    pos_weight = torch.tensor(np.sqrt((len(train) - pos) / pos), dtype=torch.float)
    MODELS_DIR.mkdir(exist_ok=True)

    for name, cfg in EXPERIMENTS.items():
        if args.only and name not in args.only:
            continue
        print(f"== {name} ({cfg['model']})")
        runs = []
        for seed in args.seeds:
            start = time.time()
            tok, model, epoch, n_trainable = train_one(cfg, seed, train, dev, pos_weight)
            dev_p = predict_proba(model, make_loader(dev, tok, False))
            test_p = predict_proba(model, make_loader(test, tok, False))
            thr = tune_thresholds(dev[EMOTIONS], dev_p)
            run = dict(seed=seed, epoch=epoch, thr=thr, test_p=test_p,
                       dev=compute_metrics(dev[EMOTIONS], dev_p >= thr),
                       test=compute_metrics(test[EMOTIONS], test_p >= thr),
                       minutes=(time.time() - start) / 60)
            print(f"  seed {seed}: best epoch {epoch}, dev {run['dev']['macro_f1']:.4f}, "
                  f"test {run['test']['macro_f1']:.4f}, {run['minutes']:.1f} min")
            if not runs or run["dev"]["macro_f1"] > max(r["dev"]["macro_f1"] for r in runs):
                out = MODELS_DIR / name  # keep the seed with the best DEV score
                (model.merge_and_unload() if cfg.get("lora") else model).save_pretrained(out)
                tok.save_pretrained(out)
                (out / "thresholds.txt").write_text(" ".join(f"{t:.2f}" for t in thr))
            runs.append(run)
            del model
            torch.cuda.empty_cache()

        mean = lambda split: {k: float(np.mean([r[split][k] for r in runs])) for k in runs[0][split]}
        std = float(np.std([r["test"]["macro_f1"] for r in runs]))
        log_experiment(name, {k: v for k, v in cfg.items()} | {
            "seeds": args.seeds, "test_macro_f1_std": round(std, 4),
            "epochs": [r["epoch"] for r in runs], "trainable_params": n_trainable,
            "minutes_per_run": round(float(np.mean([r["minutes"] for r in runs])), 1),
            "max_len": MAX_LEN, "batch": BATCH}, mean("dev"), mean("test"))
        rep = max(runs, key=lambda r: r["dev"]["macro_f1"])
        save_report(name, test, (rep["test_p"] >= rep["thr"]).astype(int), rep["test_p"])
        print(f"{name}: test macro-F1 {mean('test')['macro_f1']:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
