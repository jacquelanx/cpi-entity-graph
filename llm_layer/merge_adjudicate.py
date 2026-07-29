"""
LLM use #1: merge adjudication ("are these two mentions the same real person?").
Called by the coref stage (`graph/coref.py`) as the double-gate: a coref-linked
pair is only merged when the LLM also confirms they're the same person, on real
evidence. 
"""

from __future__ import annotations
from .llm import LLMClient, _windows


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
    Returns {same, confidence, evidence} or None if the LLM is unavailable."""
    if client is None or not client.available():
        return None
    a_forms = ", ".join(a.sorted_mentions) or "?"
    b_forms = ", ".join(b.sorted_mentions) or "?"
    prompt = (
        f'Name A: "{a_forms}"\nContexts where A appears:\n'
        + "\n".join(_windows(transcript, a))
        + f'\n\nName B: "{b_forms}"\nContexts where B appears:\n'
        + "\n".join(_windows(transcript, b))
        + "\n\nDo Name A and Name B refer to the same real person?"
    )
    return client.judge(prompt, system=_SAME_PERSON_SYSTEM)
