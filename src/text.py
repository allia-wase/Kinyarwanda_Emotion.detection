"""Word tokenizer and vocabulary for the Word2Vec and BiLSTM models.

(The Transformer models bring their own sub-word tokenizers instead.)
"""
import re
from collections import Counter

# A word with its elision apostrophe kept attached to the left part:
#   "n'abandi" -> ["n'", "abandi"]   ("na abandi" = "and the others")
# so "abandi" is shared with every other occurrence of the word instead of
# becoming a new, unseen token. Punctuation and emojis become their own tokens.
TOKEN_RE = re.compile(r"\w+'|\w+|[^\w\s]")

PAD, UNK = "<pad>", "<unk>"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


class Vocab:
    def __init__(self, tokens: list[str]):
        self.itos = [PAD, UNK] + [t for t in tokens if t not in (PAD, UNK)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_count: int = 2) -> "Vocab":
        counts = Counter(tok for text in texts for tok in tokenize(text))
        return cls([t for t, c in counts.most_common() if c >= min_count])

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = [self.stoi.get(t, 1) for t in tokenize(text)]
        return ids[:max_len] if max_len else ids

    def __len__(self):
        return len(self.itos)
