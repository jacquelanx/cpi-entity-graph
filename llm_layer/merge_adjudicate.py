"""
Merge adjudication: "are these two name mentions the same real person?"

PURPOSE
    Answer one yes/no question about a pair of person entities, with a confidence
    and a supporting quote.

FIT
    The only `llm_layer` module called from inside `graph/rules/` rather than from
    the pipeline's proposal phase. `graph/rules/coref.py` uses it as the second
    half of its double gate (a coref link only becomes a merge if the model
    confirms it) and `graph/rules/name_matching.py` uses it to VETO a containment
    merge. Either way `graph/checks/merges.py` still verifies the outcome.

HOW
    A single prompt carrying both entities' surface forms plus wide context
    windows around each, and a system prompt that pushes hard toward `false`:
    over-merging fuses two real people and cannot be undone, so the model is told
    to require explicit evidence and to treat mere name similarity as
    insufficient.

LLM use #1: merge adjudication ("are these two mentions the same real person?").
Called by the coref stage (`graph/rules/coref.py`) as the double-gate: a coref-linked
pair is only merged when the LLM also confirms they're the same person, on real
evidence. 
"""

from __future__ import annotations
from .client import LLMClient, _windows


_SAME_PERSON_SYSTEM = (
    "You are a careful de-identification assistant. You decide whether two name "
    "mentions from a single interview transcript refer to the SAME real person. "
    "Be conservative: answer same=true ONLY when the surrounding text gives clear "
    "evidence -- an explicit alias ('we called him X'), a shared full name, or an "
    "unambiguous coreference. If it is a guess or the two are simply similar "
    "names, answer same=false. Reply with ONLY a JSON object of the form "
    '{"same": true or false, "confidence": "high" or "low", '
    '"evidence": "<short quote from the text, or empty string>"}.'
)


def adjudicate_same_person(client: LLMClient | None, transcript: str, a, b) -> dict | None:
    """Ask the LLM whether entities `a` and `b` are the same person.

    Returns `{same, confidence, evidence}` or None if the LLM is unavailable --
    and the callers treat None as "no opinion", which is why a missing model
    leaves rule behaviour unchanged.

    The context windows are deliberately WIDER than the `_windows` default (300
    characters, up to 4 snippets per side): telling two same-named people apart,
    as opposed to spotting an explicit alias, needs surrounding narrative rather
    than a single keyhole around each mention.
    """
    if client is None or not client.available():
        return None
    a_forms = ", ".join(a.sorted_mentions) or "?"
    b_forms = ", ".join(b.sorted_mentions) or "?"
    # wider than the default: telling two same-named people apart (vs an explicit
    # alias) benefits from more surrounding narrative than a single ~160-char keyhole.
    prompt = (
        f'Name A: "{a_forms}"\nContexts where A appears:\n'
        + "\n".join(_windows(transcript, a, radius=300, max_snips=4))
        + f'\n\nName B: "{b_forms}"\nContexts where B appears:\n'
        + "\n".join(_windows(transcript, b, radius=300, max_snips=4))
        + "\n\nDo Name A and Name B refer to the same real person?"
    )
    return client.judge(prompt, system=_SAME_PERSON_SYSTEM)
