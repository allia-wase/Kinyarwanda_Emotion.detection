# Amarangamutima: Emotion Detection for Kinyarwanda Text

Multi-label emotion classification for Kinyarwanda: given a short text, predict which of six emotions it expresses (**anger, disgust, fear, joy, sadness, surprise**), or none.

> Work in progress. Results, the deployed app link, the demo video and the report will be added as they are completed.

| Resource | Link |
|---|---|
| Live web app | _coming soon_ (Hugging Face Spaces) |
| Fine-tuned model | _coming soon_ (`Alliane/kinyarwanda-emotion-afroxlmr`) |
| Demo video | _coming soon_ |
| Report (PDF) | _coming soon_ |

## Problem and motivation

Kinyarwanda is spoken by over 12 million people, yet it has very few NLP tools. An emotion detector is a building block for a Kinyarwanda-language wellbeing check-in tool for students. For example, it could surface sadness or fear in what someone writes and suggest support resources. **It is a support aid, not a diagnostic tool.**

## Dataset

[BRIGHTER](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories), Kinyarwanda (`kin`) subset (Muhammad et al., 2025; SemEval-2025 Task 11, Track A). Licence: CC-BY-4.0. Texts were annotated by native speakers.

| Split | Rows on Hugging Face | Unique texts used |
|---|---|---|
| train | 2,451 | 2,451 |
| dev | 814 | 407 |
| test | 2,462 | 1,231 |

**Data-quality finding:** the Hugging Face release lists every dev/test text twice (once under a Track A id, once under a Track C id) with identical labels. Copies are removed so no test text is counted twice (`src/data.py`).

Label distribution (train): sadness 26%, anger 18%, joy 17%, fear 6%, disgust 5%, surprise 4%; 29% of texts carry no emotion and 5% carry two or more.

Unlabelled text for Word2Vec: the [KINNEWS](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus) news corpus (Niyongabo et al., 2020), 11,089 articles, ~4.4M tokens. Any article containing a dev/test sentence is removed first.

## Approach

| Group | Models | Script |
|---|---|---|
| Baselines | random (by label frequency); TF-IDF (word / character n-grams) + one-vs-rest logistic regression | `src/baseline.py` |
| Neural | Word2Vec (skip-gram with negative sampling, implemented in PyTorch) → BiLSTM with max or attention pooling | `src/word2vec.py`, `src/bilstm.py` |
| Transformers | XLM-R base, AfriBERTa large, AfroXLMR base/large (full fine-tuning), AfroXLMR base + LoRA | `src/transformer.py` |

Main metric: **macro-F1 over the six emotions** (the official SemEval-2025 Task 11 metric). Each emotion's decision threshold is tuned on dev. Neural models are run with 3 seeds and reported as mean ± std.

## Repository layout

```
src/
  config.py       paths and constants
  data.py         download, normalise, deduplicate -> data/processed/
  text.py         tokenizer + vocabulary (Word2Vec / BiLSTM)
  evaluate.py     metrics, threshold tuning, confusion matrices, error files
  baseline.py     experiment group 1
  word2vec.py     Kinyarwanda Word2Vec
  bilstm.py       experiment group 2
  transformer.py  experiment group 3 (GPU)
notebooks/
  02_transformers_colab.ipynb   runs group 3 on Google Colab
results/
  experiments.csv               one row per experiment (the results table)
  <experiment>/errors.csv       every misclassified test text
  figures/                      confusion matrices
```

## Reproduce

```bash
pip install -r requirements.txt
python -m src.data          # data/processed/{train,dev,test}.csv
python -m src.baseline      # ~30 s
python -m src.word2vec      # ~35 min on CPU
python -m src.bilstm        # ~1 h on CPU
python -m src.transformer   # GPU; or open notebooks/02_transformers_colab.ipynb in Colab
```

## References

- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. *COLING 2022*.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS 2013*.
- Muhammad, S. H., et al. (2025). BRIGHTER: BRIdging the gap in human-annotated textual emotion recognition datasets for 28 languages. *ACL 2025*.
- Niyongabo, R. A., Qu, H., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. *COLING 2020*.
- Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. *MRL Workshop, EMNLP 2021*.
