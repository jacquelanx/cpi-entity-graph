"""
LLM proposer for `interviewee_identity`: which named person is the SPEAKER?

PURPOSE
    Nominate one of the detected people as the interviewee, or decline. Returns
    the `{"value": entity_id, "confidence": ...}` shape `graph.second_line`
    arbitrates -- so, like the rest of `llm_layer`, this imports nothing from
    `graph`.

FIT
    Called by `graph/rules/interviewee.resolve_interviewee_identity` as step 2 of
    that field's four steps, and gated hard by `graph/checks/interviewee.py`.

HOW
    A ROSTER prompt. Each detected person is given a short handle (P1, P2, ...)
    and the model answers with one handle or an empty string -- which is far more
    robust than asking it to echo a name back. The transcript excerpt keeps both
    ENDS of the interview, with speaker labels intact, because introductions and
    sign-offs (the two places a subject actually gets named) cluster there.

The rule layer (`graph/rules/interviewee.py`) reads three closed constructions -- the
speaker's turn label, a first-person self-introduction, an interviewer's address.
Those are high precision but far from complete: a subject can be named once, in
passing, by a construction no regex enumerates ("Boudreaux is what everybody's
called me since the mine"), or named only by the interviewer's closing thanks.

So the model gets the same question. It sees the roster of detected people and the
opening and closing stretches of the transcript with speaker labels intact --
where introductions and sign-offs live -- and returns ONE roster id or none.

It PROPOSES only. `graph/checks/interviewee.py` gates the answer: the name must
still be the speaker's label, a self-reference in a SPEAKER turn, or an
interviewer address, must not be introduced as the speaker's relative, and must
not be a public figure. A model that guesses "the first person mentioned" is
refuted, not obeyed.

Returns the `{"value": <entity_id>, "confidence": ...}` shape
`graph.second_line` arbitrates, so this module -- like the rest of `llm_layer` --
imports nothing from `graph`.
"""

from __future__ import annotations

# Introductions and sign-offs cluster at the two ends of an interview, so the
# prompt carries both ends rather than a middle slice. Bounded so one call is
# enough whatever the transcript length.
_HEAD_CHARS = 2600
_TAIL_CHARS = 1400

_SYS = (
    "You are given an interview transcript and a roster of the people named in "
    "it. Exactly one participant is the INTERVIEWEE -- the person being "
    "interviewed, who speaks in the first person ('I', 'me', 'my'). Decide "
    "whether any roster entry is the INTERVIEWEE THEMSELVES, as opposed to "
    "somebody they talk about (a relative, a neighbor, a public figure) or the "
    "interviewer.\n"
    "Answer with a roster id ONLY when the text actually names the interviewee -- "
    "they introduce themselves ('my name is ...'), the interviewer addresses them "
    "by name, or the speaker labels use their name. Many transcripts never name "
    "the interviewee at all; in that case return an empty id. A person the speaker "
    "refers to as 'my mother', 'my uncle', 'my daughter' is NEVER the interviewee. "
    "Do not guess.\n"
    'Reply with ONLY a JSON object: {"id": "<roster id such as P2, or empty>", '
    '"evidence": "<exact quote from the text, or empty>", '
    '"confidence": "high" or "low"}.'
)


def _excerpt(transcript: str) -> str:
    """The head and tail of the transcript, with the middle elided if it is long.

    A short transcript is returned whole. A long one becomes
    `head + "[... middle omitted ...]" + tail`, keeping the two regions where a
    subject is named -- the opening introductions and the closing thanks -- inside
    one bounded prompt.
    """
    if len(transcript) <= _HEAD_CHARS + _TAIL_CHARS:
        return transcript
    return (transcript[:_HEAD_CHARS].rstrip()
            + "\n\n[... middle of the interview omitted ...]\n\n"
            + transcript[-_TAIL_CHARS:].lstrip())


def propose_interviewee(client, transcript: str, persons: list, interviewee) -> dict | None:
    """Return `{"value": entity_id, "confidence": ...}` or None.

    None means "no proposal" -- which the second line records as the LLM having no
    answer, distinct from a proposal the checkers refuted.

    HOW: build the `P1 = Aunt Maria` roster, ask, then translate the answer back.
    The reply is normalized defensively -- uppercased, and a bare "2" is repaired
    to "P2" -- and an id that is not in the roster returns None rather than
    guessing. The evidence quote is truncated to 200 characters, since it is
    provenance for a reviewer, not part of the decision.
    """
    if client is None or not client.available() or not persons:
        return None

    roster = {f"P{i}": e for i, e in enumerate(persons, start=1)}
    lines = "\n".join(
        f"{pid} = {e.sorted_mentions[0] if e.sorted_mentions else e.entity_id}"
        for pid, e in roster.items())
    prompt = (f"Roster of named people:\n{lines}\n\n"
              f"Transcript (speaker labels kept):\n{_excerpt(transcript)}\n\n"
              "Which roster id, if any, is the interviewee themselves?")
    res = client.judge(prompt, system=_SYS)
    if not res:
        return None

    pid = str(res.get("id") or "").strip().upper()
    if not pid:
        return None
    if not pid.startswith("P"):
        pid = "P" + pid.lstrip("Pp")
    ent = roster.get(pid)
    if ent is None:
        return None
    return {"value": ent.entity_id,
            "confidence": str(res.get("confidence") or "").strip().lower() or "unstated",
            "evidence": str(res.get("evidence") or "")[:200]}
