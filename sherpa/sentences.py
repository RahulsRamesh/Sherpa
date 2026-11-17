"""Incremental sentence splitting for text that arrives in small deltas.

Feed it Claude's streamed text; it returns each sentence the moment it is
certain, so TTS can start on sentence one while sentence two is still being
written. A sentence is certain once its terminal punctuation is followed by
whitespace (and, for a period, by the first letter of what comes next, which
rules out "e.g. this" and "approx. five")."""

import re
from typing import List, Optional

MIN_CHARS = 16  # shorter sentences ("Sure.", "Yes.") merge into the next one
LONG_RUN_CHARS = 200  # no sentence end for this long: break at the next comma/clause
HARD_LIMIT_CHARS = 400  # ...and if there isn't even a comma, at the next space

# punctuation run, optional closing quotes/brackets, whitespace, then the next character if it has arrived
_END = re.compile(r"([.!?]+)([\"'”’)\]]*)(\s+)(\S)?")
_CLAUSE = re.compile(r"[,;:]\s|\s[—–-]+\s")
_LAST_WORD = re.compile(r"(\S+)$")
_INITIALS = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")  # e.g, i.e, U.S, a.m
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "approx", "fig", "no", "inc", "ltd",
    "co", "dept", "est", "min", "max", "ver", "vol", "cf", "al", "eg", "ie",
}


class SentenceSplitter:
    def __init__(self, min_chars: int = MIN_CHARS, long_run: int = LONG_RUN_CHARS):
        self.min_chars = min_chars
        self.long_run = long_run
        self._buf = ""

    def feed(self, text: str) -> List[str]:
        """Add a delta; return the sentences that are now complete (maybe none)."""
        self._buf += text
        out = []
        while True:
            sentence = self._extract()
            if sentence is None:
                break
            out.append(sentence)
        return out

    def flush(self) -> List[str]:
        """End of stream: whatever is left is the last sentence."""
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []

    def _extract(self) -> Optional[str]:
        buf = self._buf
        for m in _END.finditer(buf):
            punct = m.group(1)
            if set(punct) == {"."} and len(punct) > 1:
                continue  # ellipsis: a pause, not an end
            if punct == "." and not self._period_ends_sentence(buf, m):
                if m.group(4) is None:
                    break  # can't tell yet; wait for the next character
                continue
            candidate = buf[: m.end(2)].strip()
            if len(candidate) < self.min_chars:
                continue
            self._buf = buf[m.end(3):]
            return candidate
        return self._extract_long_run()

    def _period_ends_sentence(self, buf: str, m: "re.Match") -> bool:
        word_m = _LAST_WORD.search(buf[: m.start(1)])
        word = word_m.group(1).lstrip("(\"'[“") if word_m else ""
        if _INITIALS.fullmatch(word) or word.lower() in ABBREVIATIONS:
            return False
        if word.isdigit() and len(word) <= 2:
            return False  # "1. Open the menu" (numbered list)
        if len(word) == 1 and word.isalpha() and word != "I":
            return False  # an initial: "J. Smith"
        nxt = m.group(4)
        if nxt is None:
            return False  # wait; the caller sees group(4) is None
        return not nxt.islower()

    def _extract_long_run(self) -> Optional[str]:
        buf = self._buf
        if len(buf) < self.long_run:
            return None
        for m in _CLAUSE.finditer(buf, self.long_run - 1):
            piece = buf[: m.start() + (1 if buf[m.start()] in ",;:" else 0)].strip()
            self._buf = buf[m.end():]
            return piece
        if len(buf) >= HARD_LIMIT_CHARS:
            cut = buf.rfind(" ", 0, HARD_LIMIT_CHARS)
            if cut > 0:
                self._buf = buf[cut + 1:]
                return buf[:cut].strip()
        return None
