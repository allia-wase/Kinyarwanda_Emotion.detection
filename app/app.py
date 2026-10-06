"""Amarangamutima: Kinyarwanda emotion detection web app (Gradio, Hugging Face Spaces).

How the app connects to the model:
  1. At start-up it downloads the fine-tuned AfroXLMR model, its tokenizer and
     the per-emotion thresholds (thresholds.txt) from the Hugging Face Hub.
  2. A user's text goes through the SAME normalisation used in training
     (normalize() below is a copy of src/data.py:normalize).
  3. The model returns one sigmoid probability per emotion; an emotion is
     predicted when its probability passes that emotion's tuned threshold.
"""
import os
import re
import unicodedata

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Alliane/kinyarwanda-emotion-afroxlmr")
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
KIN = {"anger": "Uburakari", "disgust": "Isesemi", "fear": "Ubwoba", "joy": "Ibyishimo",
       "sadness": "Agahinda", "surprise": "Gutungurwa"}
EMOJI = {"anger": "😠", "disgust": "🤢", "fear": "😨", "joy": "😊", "sadness": "😢", "surprise": "😲"}

APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    """Identical to src/data.py:normalize. Keep the two in sync."""
    text = unicodedata.normalize("NFKC", str(text)).translate(APOSTROPHES)
    text = re.sub(r"@<username>|@\w+", " @user ", text)
    text = re.sub(r"##url##|https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
try:
    with open(hf_hub_download(MODEL_ID, "thresholds.txt")) as f:
        THRESHOLDS = [float(x) for x in f.read().split()]
except Exception:  # model uploaded without thresholds: fall back to 0.5
    THRESHOLDS = [0.5] * len(EMOTIONS)


@torch.no_grad()
def predict(text: str):
    text = normalize(text or "")
    if not text:
        return "Andika interuro mu Kinyarwanda. (Type a sentence in Kinyarwanda.)", {}, ""
    enc = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    probs = torch.sigmoid(model(**enc).logits)[0].tolist()

    found = [e for e, p, t in zip(EMOTIONS, probs, THRESHOLDS) if p >= t]
    if found:
        verdict = "  ".join(f"{EMOJI[e]} **{KIN[e]}** ({e})" for e in found)
    else:
        verdict = "😐 **Nta marangamutima yihariye** (no clear emotion detected)"

    # Each emotion's score relative to its own threshold: >= 1.0 means "predicted".
    scores = {f"{EMOJI[e]} {e}": round(p, 3) for e, p in zip(EMOTIONS, probs)}
    detail = "| Emotion | Probability | Threshold | Predicted |\n|---|---|---|---|\n" + "\n".join(
        f"| {e} | {p:.3f} | {t:.2f} | {'✅' if p >= t else ''} |"
        for e, p, t in zip(EMOTIONS, probs, THRESHOLDS))
    detail += f"\n\nTokens seen by the model: `{' '.join(tokenizer.tokenize(text))}`"
    return verdict, scores, detail


EXAMPLES = [
    # From the BRIGHTER test set (gold label in the comment)
    ["uwo mwana nta mubyeyi agira kuko bose bapfuye."],                                   # sadness
    ["ibimeme bye biranuka wagirango ntazi koga"],                                        # disgust
    ["ntabwo twari tubimenyereye ariko twabonye twarakinnye neza."],                      # joy
    ["abasirikare bamuhiritse ku butegetsi banze kuva ku izima, bavuga ko batazabumusubizaho."],  # anger
    ["kuko nihatabaho kubirengera bizatuma ubuzima bugenda nabi bityo amahoro ntakomeze kubaho."],  # fear
    ["ubu bukwe bwabaye mu ibanga rikomeye dore ko aba bombi nta mpapuro bigeze batanga zisaba gushyingiranwa mu buryo bwemewe n'amategeko."],  # surprise
    ["umuntu akubaka ibitanda bine, bitanu ndetse hari n'uwakodeshaga isambu ya mugenzi we kugira ngo abone aho yanika."],  # none
    # Written for the demo (student wellbeing check-in style)
    ["ndumva mfite agahinda kenshi uyu munsi, nta muntu nshobora kubwira."],
    ["nishimiye cyane ko natsinze ikizamini!"],
]

with gr.Blocks(title="Amarangamutima: Kinyarwanda emotion detection") as demo:
    gr.Markdown(
        "# Amarangamutima 💬\n"
        "**Kinyarwanda emotion detection.** Write a sentence in Kinyarwanda and the model "
        "detects which emotions it expresses: anger, disgust, fear, joy, sadness, surprise "
        "(several, or none).\n\n"
        "Model: AfroXLMR-base fine-tuned on the Kinyarwanda part of the BRIGHTER dataset "
        "(SemEval-2025 Task 11). [Code and report on GitHub]"
        "(https://github.com/allia-wase/Kinyarwanda_Emotion.detection)")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Interuro mu Kinyarwanda (Kinyarwanda text)", lines=4,
                             placeholder="urugero: ndumva mfite ibyishimo byinshi uyu munsi")
            btn = gr.Button("Sesengura (Analyse)", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column():
            verdict = gr.Markdown()
            bars = gr.Label(label="Probability per emotion", num_top_classes=6)
            with gr.Accordion("How the model decided", open=False):
                detail = gr.Markdown()
    gr.Markdown(
        "⚠️ **Research prototype, not a diagnostic tool.** It can be wrong, especially with "
        "sarcasm, proverbs, slang or mixed languages. If you or someone you know is "
        "struggling, please talk to someone you trust or a health professional.")
    btn.click(predict, inp, [verdict, bars, detail])
    inp.submit(predict, inp, [verdict, bars, detail])

if __name__ == "__main__":
    demo.launch()
