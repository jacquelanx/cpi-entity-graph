"""
Deterministic sentence segmentation shared by the graph stages.

A single robust splitter that returns (start, end) character spans tiling the
whole transcript. It splits on '.', '?', '!' but NOT on the periods the naive
`[.!?]` scanner used to break on:

  * honorific / common abbreviations : "Dr. Nguyen", "Mr. Kim", "etc."
  * single-letter initials / acronyms: "J. Smith", "U.S."
  * decimal numbers                  : "3.14"
  * ellipses                         : "well... okay"

The spans tile the text contiguously (each starts where the previous ended, the
last ends at len(text)), so `sentence_of(pos)` finds a containing span for any
0 <= pos < len(text). Leading whitespace belongs to the following span; callers
strip() before display.
"""

from __future__ import annotations
import re

# Lowercased tokens whose trailing period does NOT end a sentence.
_ABBREV = {
    # personal / professional titles
    "mr", "mrs", "ms", "mx", "dr", "prof", "st", "sr", "jr", "rev", "fr",
    "hon", "gov", "sen", "rep", "pres", "capt", "sgt", "lt", "col", "gen",
    "cmdr", "cpl", "det", "ofc", "supt", "atty", "messrs", "mmes",
    # generic / latin
    "etc", "vs", "al", "inc", "ltd", "co", "corp", "dept", "est", "fig",
    "no", "nos", "vol", "pp", "approx", "apt", "ave", "blvd", "rd", "ste",
    "cf", "viz", "ibid",
}

_WORD_BEFORE_DOT = re.compile(r"([A-Za-z][A-Za-z'&.]*)$")
_INITIALISM = re.compile(r"^[a-z](?:\.[a-z])+$")     # u.s, e.g, i.e
_TERMINATORS = ".?!"
_CLOSERS = "\"'”’)]"                       # quotes/brackets that trail a terminator


def _ends_sentence(text: str, seg_start: int, dot: int, run: str) -> bool:
    """True if the terminator run [dot:dot+len(run)] really ends a sentence."""
    if run != ".":
        # '?', '!', '?!' end a sentence; a run of 2+ dots is an ellipsis and does not
        return set(run) != {"."}
    n = len(text)
    if 0 < dot < n - 1 and text[dot - 1].isdigit() and text[dot + 1].isdigit():
        return False                                 # decimal number: 3.14
    wb = _WORD_BEFORE_DOT.search(text[seg_start:dot])
    if wb:
        tok = wb.group(1).lower()
        if tok in _ABBREV or len(tok) == 1 or _INITIALISM.match(tok):
            return False                             # abbreviation / initial / acronym
    return True


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans that tile `text`; see the module docstring."""
    n = len(text)
    spans: list[tuple[int, int]] = []
    start = i = 0
    while i < n:
        if text[i] in _TERMINATORS:
            j = i
            while j < n and text[j] in _TERMINATORS:  # consume "?!", "...", "!!!"
                j += 1
            if _ends_sentence(text, start, i, text[i:j]):
                end = j
                while end < n and text[end] in _CLOSERS:
                    end += 1
                spans.append((start, end))
                start = i = end
                continue
            i = j
            continue
        i += 1
    if start < n:
        spans.append((start, n))
    return spans
