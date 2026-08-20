"""
The simulated perfect detector.

PURPOSE
    Turn the gold annotations into the detection spans `graph.pipeline` expects,
    so the harness measures THIS stage rather than the detector in front of it.

FIT
    Called by `scoring.evaluate_one`. The sibling implementation for the HTML
    reports is `demo/cases.detections_from_gold`; the two are kept deliberately
    parallel.

HOW
    For every gold surface form, find EVERY occurrence of it in the transcript and
    emit a detection span with score 1.0. So a perfect detector, by construction:
    it finds all the right spans and nothing else.
"""

from __future__ import annotations

import re


def _find_spans(text: str, surface: str):
    """Every occurrence of `surface` in `text`, at letter boundaries.

    The `(?<![A-Za-z])` / `(?![A-Za-z])` lookarounds require a non-letter on each
    side, so "Ruth" does not match inside "Ruthie". Written this way rather than
    with `\b` because a surface form may contain spaces or punctuation, where a
    word-boundary assertion would land in the wrong place. `re.escape` keeps a
    name containing regex characters (a period, a hyphen) from being treated as a
    pattern.
    """
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(surface) + r"(?![A-Za-z])")
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def _build_detections(text: str, gold: dict):
    """Build the full detection list for one transcript from its gold annotations.

    Walks each gold section in turn -- people, the interviewee's own name,
    locations, dates (each carrying its own DATE_* category), ages and direct
    identifiers -- and adds one detection per occurrence via the local `add`
    helper.

    The interviewee's forms are kept in a SEPARATE gold section from `people`, so
    that clustering and relation scoring continue to measure third parties only,
    while the speaker's name is still detected when the transcript states it.
    """
    dets = []

    def add(surface, etype):
        """Emit one detection per occurrence of `surface`, typed `etype`."""
        for s, e in _find_spans(text, surface):
            dets.append({"text": surface, "start": s, "end": e,
                         "entity_type": etype, "score": 1.0})

    for p in gold["people"]:
        for form in p["forms"]:
            add(form, "PERSON")
    # The speaker's own name, when the transcript states it -- see the same note in
    # `demo.cases.detections_from_gold`. Kept out of `people` so clustering and
    # relation scoring still describe THIRD PARTIES only.
    for form in (gold.get("interviewee") or {}).get("forms", []):
        add(form, "PERSON")
    for loc in gold.get("locations", []):
        add(loc["text"], "LOCATION")
    for d in gold.get("dates", []):
        add(d["text"], d["category"])
    for a in gold.get("ages", []):
        add(a["text"], "AGE")
    # Direct identifiers were previously omitted, so the harness never built any
    # identifier entities -- which meant no ownership resolution and no identifier
    # rows in the second-line ledger.
    for x in gold.get("identifiers", []):
        add(x["text"], x["type"])
    return dets
