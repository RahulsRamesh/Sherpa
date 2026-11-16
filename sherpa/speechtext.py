"""Compare what Sherpa said with what the mic heard, despite how numbers differ.

The text we send to TTS says "$1,200" or "2026"; the speech-to-text of the
echo says "twelve hundred dollars" or "twenty twenty six" (or the reverse).
Both sides are reduced to spoken words so they can be compared.
"""

import difflib
import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD = {"first": "one", "second": "two", "third": "three", "fifth": "five", "eighth": "eight", "ninth": "nine", "twelfth": "twelve"}


def _below_100(n: int) -> list:
    if n < 20:
        return [_ONES[n]]
    return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])


def _below_1000(n: int) -> list:
    out = []
    if n >= 100:
        out += [_ONES[n // 100], "hundred"]
        n %= 100
    if n:
        out += _below_100(n)
    return out


def cardinal(n: int) -> list:
    if n == 0:
        return ["zero"]
    out = []
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= size:
            out += _below_1000(n // size) + [name]
            n %= size
    if n:
        out += _below_1000(n)
    return out


def _variants(n: int) -> list:
    """Ways a number might be spoken. The first is the canonical one."""
    v = [cardinal(n)]
    if 1100 <= n <= 2999 and n % 100:  # years: "twenty twenty six"
        v.append(cardinal(n // 100) + (["oh"] if n % 100 < 10 else []) + cardinal(n % 100))
    if n >= 100 and n % 100 == 0 and n < 10000:  # "twelve hundred"
        v.append(cardinal(n // 100) + ["hundred"])
    return v


_TOKEN = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?(?:st|nd|rd|th)?|[a-z']+")


def _expand(token: str) -> list:
    """Variants (lists of words) for one token."""
    if not token[0].isdigit() and token[0] != "$":
        return [[_ORD.get(token, token)]]
    money, pct = token.startswith("$"), token.endswith("%")
    core = re.sub(r"(st|nd|rd|th)$", "", token.strip("$%")).replace(",", "")
    try:
        whole, _, frac = core.partition(".")
        n = int(whole or 0)
    except ValueError:
        return [[token]]
    out = []
    for v in _variants(n):
        v = list(v)
        if frac:
            v += ["point"] + [_ONES[int(d)] for d in frac]
        if money:
            v += ["dollars"]
        if pct:
            v += ["percent"]
        out.append(v)
    return out


def spoken_words(text: str) -> list:
    """Canonical spoken form: one word list, numbers spelled out."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        out += _expand(tok)[0]
    return out


def spoken_vocabulary(text: str) -> set:
    """Every word that might be heard when `text` is read aloud: all variants of each number."""
    vocab = set()
    for tok in _TOKEN.findall(text.lower()):
        for v in _expand(tok):
            vocab.update(v)
    if "$" in text:
        vocab.update({"dollar", "bucks"})
    return vocab


def _hit(w: str, vocab: set) -> bool:
    return w in vocab or (len(w) > 3 and bool(difflib.get_close_matches(w, vocab, n=1, cutoff=0.8)))


def overlap(heard: str, spoken: str) -> float:
    """Fraction of heard words that plausibly came from `spoken`; tolerant of
    small transcription differences and of numbers written either way."""
    vocab = spoken_vocabulary(spoken)
    hits = total = 0
    for tok in _TOKEN.findall(heard.lower()):
        # A number can be spoken several ways: score it by its best-matching one.
        best = max(((sum(_hit(w, vocab) for w in v), len(v)) for v in _expand(tok)), key=lambda hv: hv[0] / hv[1])
        hits += best[0]
        total += best[1]
    return hits / total if total else 0.0
