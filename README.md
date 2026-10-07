# Amarangamutima: Emotion Detection for Kinyarwanda Text

Multi-label emotion classification for Kinyarwanda: given a short text, predict which of six emotions it expresses (**anger, disgust, fear, joy, sadness, surprise**), or none.

| Resource | Link |
|---|---|
| Live web app | _coming soon_ (Streamlit Community Cloud) |
| Fine-tuned model | [Alliane/kinyarwanda-emotion-afroxlmr](https://huggingface.co/Alliane/kinyarwanda-emotion-afroxlmr) |
| Demo video | _coming soon_ |
| Report (PDF) | _coming soon_ |

**Best model:** AfroXLMR-large fine-tuned on 2,451 Kinyarwanda texts: **test macro-F1 0.592 ± 0.006** (3 seeds), against 0.511 for the TF-IDF baseline and 0.363 for XLM-R, which never saw Kinyarwanda in pretraining.

## Problem and motivation

Kinyarwanda, the national language of Rwanda, has very few NLP tools. An emotion detector is a building block for a Kinyarwanda-language wellbeing check-in tool for students. For example, it could surface sadness or fear in what someone writes and suggest support resources. **It is a support aid, not a diagnostic tool.**

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

## Results

Test set, 1,231 texts. Neural models: mean ± std over 3 seeds. Model selection uses **dev** only.

| Exp | Model | Dev macro-F1 | Test macro-F1 | Test micro-F1 |
|---|---|---|---|---|
| B0 | Random, by label frequency | 0.140 | 0.124 | 0.166 |
| B1 | Word TF-IDF + LogReg, threshold 0.5 | 0.198 | 0.208 | 0.348 |
| B2 | + class weights + tuned thresholds | 0.486 | 0.466 | 0.492 |
| B3 | Character TF-IDF + LogReg (tuned) | 0.524 | 0.504 | 0.513 |
| B4 | Word + character TF-IDF + LogReg (tuned) | 0.534 | 0.511 | 0.509 |
| L1 | BiLSTM, random embeddings | 0.452 | 0.413 ± 0.008 | |
| L2 | BiLSTM, Word2Vec (frozen) | 0.518 | 0.465 ± 0.003 | |
| L3 | BiLSTM, Word2Vec (fine-tuned) | 0.511 | 0.478 ± 0.007 | |
| L4 | BiLSTM, Word2Vec + attention pooling | 0.516 | 0.458 ± 0.009 | |
| T1 | XLM-R base (no Kinyarwanda in pretraining) | 0.395 | 0.363 ± 0.050 | 0.353 |
| T2 | AfriBERTa large | 0.545 | 0.508 ± 0.004 | |
| T3 | AfroXLMR base | 0.585 | 0.538 ± 0.003 | 0.569 |
| T5 | AfroXLMR base + LoRA (0.4% of weights trained) | 0.588 | 0.525 ± 0.018 | 0.558 |
| **T4** | **AfroXLMR large** | **0.614** | **0.592 ± 0.006** | |

Full table with every metric and configuration: [`results/experiments.csv`](results/experiments.csv).

**What we learned**
1. **Handling imbalance is the biggest single win** (B1 → B2: +0.26). With a 0.5 threshold, fear and surprise are never predicted.
2. **Seeing Kinyarwanda in pretraining matters most.** XLM-R and AfroXLMR-base share architecture and tokenizer. AfroXLMR's extra African-language pretraining is worth +0.17 macro-F1, and XLM-R was also unstable (one seed collapsed after epoch 1).
3. **More complex is not automatically better.** The BiLSTM never beats TF-IDF on 2.4k texts. Pretrained Word2Vec helps it (+0.065, test OOV 24.6% → 9.4%), attention pooling does not.
4. **Scale still helps with little data:** AfroXLMR-large beats base by +0.05, on all three seeds.
5. **LoRA** keeps ~98% of full fine-tuning's score while training 0.4% of the weights, about 3× faster, but varies more across seeds.

## Error analysis

`python -m src.error_analysis` → [`results/error_analysis/`](results/error_analysis/)

| Emotion (test F1) | TF-IDF | BiLSTM | XLM-R | AfriBERTa | AfroXLMR-base | AfroXLMR-large |
|---|---|---|---|---|---|---|
| anger | 0.49 | 0.46 | 0.47 | 0.52 | 0.55 | **0.58** |
| disgust | 0.85 | 0.77 | 0.71 | **0.87** | 0.83 | 0.85 |
| fear | 0.36 | 0.29 | 0.15 | 0.41 | **0.46** | 0.45 |
| joy | 0.64 | 0.60 | 0.48 | 0.61 | 0.66 | **0.74** |
| sadness | 0.55 | 0.55 | 0.48 | 0.57 | 0.61 | **0.69** |
| surprise | 0.17 | 0.18 | 0.11 | 0.11 | 0.13 | **0.24** |

- **Missed emotions are the most common error**: an emotional text is predicted as "none". Sadness and anger are confused in both directions; they also co-occur most in the training labels.
- **Disgust is easy for a dataset reason**: disgust texts average 7 words (vs ~18), are almost never tweets, and use distinctive vocabulary (*umwanda*, *isesemi*, *umunuko*), a shortcut the models exploit.
- **Surprise is the hardest emotion** (≤ 0.24): only 105 training examples, and it depends on what the reader expected rather than on specific words.
- **Tweets are harder than news-style sentences** (AfroXLMR-large: 0.39 vs 0.58). Jokes with 😂 and "?" are often predicted as anger, the same shortcut the TF-IDF model's top anger features reveal.
- XLM-R without Kinyarwanda **floods rare classes**: it predicted fear 508 times for 72 true cases.
- 268 test texts are wrong for all six analysed models. 21 of them were reviewed by a native speaker ([`native_speaker_review.csv`](results/error_analysis/native_speaker_review.csv)): debatable labels, sarcasm, idioms (*kwica inyoni ebyiri n'ibuye rimwe*), missing context.

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
  error_analysis.py  per-emotion, confusion, slice and hard-case analysis
  inference.py    the deployed classifier (shared by the app and predict.py)
  predict.py      command-line prediction and test-set re-scoring
app/
  streamlit_app.py   web app
notebooks/
  02_transformers_colab.ipynb   runs group 3 on Google Colab
results/
  experiments.csv               one row per experiment (the results table)
  <experiment>/errors.csv       every misclassified test text
  error_analysis/               error analysis outputs
  figures/                      confusion matrices, per-emotion F1
```

## Reproduce

```bash
pip install -r requirements.txt
python -m src.data          # data/processed/{train,dev,test}.csv
python -m src.baseline      # ~30 s
python -m src.word2vec      # ~35 min on CPU
python -m src.bilstm        # ~1 h on CPU
python -m src.transformer   # GPU; or open notebooks/02_transformers_colab.ipynb in Colab
python -m src.error_analysis
python -m src.predict "ndumva mfite agahinda kenshi uyu munsi"   # uses the published model
streamlit run app/streamlit_app.py                                # the web app, locally
```

## References

- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. *COLING 2022*.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS 2013*.
- Mosbach, M., Andriushchenko, M., & Klakow, D. (2021). On the stability of fine-tuning BERT: Misconceptions, explanations, and strong baselines. *ICLR 2021*.
- Muhammad, S. H., et al. (2025). BRIGHTER: BRIdging the gap in human-annotated textual emotion recognition datasets for 28 languages. *ACL 2025*.
- Niyongabo, R. A., Qu, H., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. *COLING 2020*.
- Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. *MRL Workshop, EMNLP 2021*.
