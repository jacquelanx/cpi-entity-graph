"""
Deterministic sentence segmentation shared by the graph stages.

PURPOSE
    Split a transcript into sentence spans. Almost every rule and checker in the
    pipeline reasons over "the sentence containing this mention" -- an ownership
    cue, a kin word, a date sitting beside an age -- so where a sentence starts
    and ends decides what evidence a rule can see.

FIT
    A leaf utility with NO knowledge of the graph: it takes text and returns
    offsets. Used by `graph/checks/__init__.CheckContext`,
    `graph/checks/relation_evidence.py`, `graph/rules/coref.py`,
    `graph/rules/ages.py` and `graph/rules/interviewee.py`. Callers typically
    wrap it in their own small `sentence_of(pos)` index lookup.

HOW
    A hand-written single-pass scanner rather than a general NLP model, so the
    output is fast, dependency-free and identical on every run. It walks the text
    once, and at each run of terminator characters asks `_ends_sentence` whether
    that run is a real boundary or a false one.

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
    """True if the terminator run at `text[dot:dot+len(run)]` really ends a sentence.

    `run` is the whole consecutive stretch of `.?!` characters found at `dot`, and
    `seg_start` is where the current sentence began (needed to look back at the
    word before the dot without straying into the previous sentence).

    The decision, in order:

      * A run that is not a single "." is easy. "?" / "!" / "?!" always end a
        sentence; a run of two or more dots is an ELLIPSIS ("well... okay") and
        does not. `set(run) != {"."}` is the test: the set of distinct characters
        equals `{"."}` exactly when the run is dots and nothing else, so anything
        containing a ? or ! ends the sentence and a pure dot-run does not.
      * A single "." between two digits is a decimal point -- "3.14" is one token,
        not two sentences.
      * Otherwise, look at the word immediately before the dot and suppress the
        break if it is a known abbreviation ("Dr", "etc"), a single letter (the
        "J." of "J. Smith"), or a dotted initialism ("u.s", "e.g").

    Anything else is a genuine sentence end.
    """
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
    """Split `text` into `(start, end)` character spans, one per sentence.

    The spans TILE the text: each begins exactly where the previous one ended and
    the last reaches `len(text)`, so every character belongs to exactly one span
    and a lookup for any position always succeeds. Leading whitespace belongs to
    the span that follows it, so callers `strip()` before displaying.

    HOW: a single left-to-right scan with two cursors -- `start` marks the
    beginning of the sentence being built, `i` is the read head. On hitting a
    terminator character, `j` runs forward to consume the ENTIRE run of them
    ("?!", "...", "!!!") so the run can be judged as a unit. If
    `_ends_sentence` accepts it, the span is closed after also absorbing any
    trailing quote or bracket (so `He said "stop!"` keeps its closing quote), and
    both cursors jump past it. If the run is rejected -- an abbreviation's period,
    an ellipsis -- the read head skips to `j` and the sentence keeps growing.

    The final `if start < n` closes a trailing sentence that ended without any
    punctuation at all, which is common in transcribed speech.
    """
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
