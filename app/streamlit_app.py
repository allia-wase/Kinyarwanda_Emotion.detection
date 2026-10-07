"""Amarangamutima: Kinyarwanda emotion detection web app (Streamlit Community Cloud).

Run locally:  streamlit run app/streamlit_app.py

How the app connects to the model:
  1. On first visit it downloads the fine-tuned AfroXLMR-large model, tokenizer
     and per-emotion thresholds from the Hugging Face Hub
     (Alliane/kinyarwanda-emotion-afroxlmr) and keeps them in memory.
  2. Text goes through src/inference.py: the same normalize() used for training,
     the model's tokenizer, the Transformer, a sigmoid per emotion.
  3. An emotion is shown as detected when its probability passes that emotion's
     threshold (tuned on the dev set during training).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `src`

import streamlit as st

from src.config import EMOTIONS
from src.inference import EmotionClassifier

KIN = {"anger": "Uburakari", "disgust": "Isesemi", "fear": "Ubwoba", "joy": "Ibyishimo",
       "sadness": "Agahinda", "surprise": "Gutungurwa"}
EMOJI = {"anger": "😠", "disgust": "🤢", "fear": "😨", "joy": "😊", "sadness": "😢", "surprise": "😲"}
EXAMPLES = {
    # From the BRIGHTER test set (gold label in brackets)
    "Umwana w'impfubyi [sadness]": "uwo mwana nta mubyeyi agira kuko bose bapfuye.",
    "Impumuro mbi [disgust]": "ibimeme bye biranuka wagirango ntazi koga",
    "Twakinnye neza [joy]": "ntabwo twari tubimenyereye ariko twabonye twarakinnye neza.",
    "Abasirikare banze [anger]": "abasirikare bamuhiritse ku butegetsi banze kuva ku izima, bavuga ko batazabumusubizaho.",
    "Amahoro mu kaga [fear]": "kuko nihatabaho kubirengera bizatuma ubuzima bugenda nabi bityo amahoro ntakomeze kubaho.\"",
    "Ubukwe bw'ibanga [surprise]": "ubu bukwe bwabaye mu ibanga rikomeye dore ko aba bombi nta mpapuro bigeze batanga zisaba gushyingiranwa mu buryo bwemewe n'amategeko.",
    "Inkuru isanzwe [none]": "umuntu akubaka ibitanda bine, bitanu ndetse hari n'uwakodeshaga isambu ya mugenzi we kugira ngo abone aho yanika.",
    "Igitekerezo cy'urwenya [none, often misread as anger]": "@user ndashaka kuzareba umuntu uha ruswa ngo ujue mu ijuru 😂😂😂😂😂😂",
    # Written for the demo (student wellbeing check-in style)
    "Check-in: agahinda": "ndumva mfite agahinda kenshi uyu munsi, nta muntu nshobora kubwira.",
    "Check-in: ibyishimo": "nishimiye cyane ko natsinze ikizamini!",
}

st.set_page_config(page_title="Amarangamutima", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="Loading the model (the first visit can take 1-2 minutes)...")
def load_model() -> EmotionClassifier:
    # bfloat16 halves memory (2.2 GB -> 1.1 GB) to fit the free hosting tier;
    # src/predict.py --evaluate checks this gives the same test results as float32.
    return EmotionClassifier(dtype=os.environ.get("MODEL_DTYPE", "bfloat16"))


st.title("💬 Amarangamutima")
st.markdown(
    "**Kinyarwanda emotion detection.** Write a sentence in Kinyarwanda and the model detects "
    "which emotions it expresses: anger, disgust, fear, joy, sadness, surprise (several, or none).")

if "text" not in st.session_state:
    st.session_state.text = ""


def use_example():
    choice = st.session_state.example
    if choice in EXAMPLES:
        st.session_state.text = EXAMPLES[choice]


st.selectbox("Hitamo urugero (pick an example)", ["-"] + list(EXAMPLES), key="example",
             on_change=use_example)
text = st.text_area("Interuro mu Kinyarwanda (Kinyarwanda text)", key="text", height=110,
                    placeholder="urugero: ndumva mfite ibyishimo byinshi uyu munsi")

if st.button("Sesengura (Analyse)", type="primary") and text.strip():
    clf = load_model()
    result = clf.predict(text)
    found = [e for e in EMOTIONS if result[e]["predicted"]]

    if found:
        st.subheader("  ".join(f"{EMOJI[e]} {KIN[e]} ({e})" for e in found))
    else:
        st.subheader("😐 Nta marangamutima yihariye (no clear emotion detected)")

    for e in EMOTIONS:
        r = result[e]
        left, right = st.columns([1, 3])
        left.markdown(f"{EMOJI[e]} **{e}**" + (" ✅" if r["predicted"] else ""))
        right.progress(min(r["probability"], 1.0),
                       text=f"{r['probability']:.2f}  (threshold {r['threshold']:.2f})")

    with st.expander("How the model decided"):
        st.markdown(
            "The model gives each emotion an independent probability (sigmoid output). An emotion "
            "is detected when its probability passes its own **threshold**, tuned on the "
            "development set: rare emotions get lower thresholds so they are not always missed.")
        st.markdown("**Sub-word tokens the model saw:**")
        st.code(" ".join(clf.tokens(text)), language=None)

st.divider()
st.caption(
    "⚠️ Research prototype, **not a diagnostic tool**. It can be wrong, especially with sarcasm, "
    "jokes, idioms, slang or mixed languages. If you or someone you know is struggling, please "
    "talk to someone you trust or a health professional.")
st.caption(
    "Model: AfroXLMR-large (Alabi et al., 2022) fine-tuned on the Kinyarwanda part of BRIGHTER "
    "(Muhammad et al., 2025; SemEval-2025 Task 11). Test macro-F1 ≈ 0.59. "
    "[Code, experiments and report](https://github.com/allia-wase/Kinyarwanda_Emotion.detection)")
