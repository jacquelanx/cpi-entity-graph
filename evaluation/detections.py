"""
The simulated perfect detector.

Turns gold surface forms into detection spans, so the harness measures THIS
stage rather than the detector in front of it.
"""

from __future__ import annotations

import re


def _find_spans(text: str, surface: str):
    """All occurrences of `surface` at letter boundaries."""
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(surface) + r"(?![A-Za-z])")
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def _build_detections(text: str, gold: dict):
    """Simulated perfect detector: gold surface forms -> detection dicts."""
    dets = []

    def add(surface, etype):
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
