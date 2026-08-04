"""
Simulates a perfect detector from the gold annotations in tests/gold/*.json and
runs the transcript through the real graph pipeline.
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.loader import resolve_overlaps, make_mentions
from graph.pipeline import run_pipeline

TESTS = REPO_ROOT / "tests"
DATA_GAZ = REPO_ROOT / "data" / "gazetteer.csv"


def all_tids():
    return [f"interview_{i:03d}" for i in range(1, 6)]


"""Turn gold surface forms into detections (a stand-in perfect detector)."""
def detections_from_gold(text: str, gold: dict):
    dets = []

    def add(surface, etype):
        for m in re.finditer(r"(?<![A-Za-z])" + re.escape(surface) + r"(?![A-Za-z])", text):
            dets.append({"text": surface, "start": m.start(), "end": m.end(),
                         "entity_type": etype, "score": 1.0})

    for p in gold["people"]:
        for f in p["forms"]:
            add(f, "PERSON")
    for l in gold.get("locations", []):
        add(l["text"], "LOCATION")
    for d in gold.get("dates", []):
        add(d["text"], d["category"])
    for a in gold.get("ages", []):
        add(a["text"], "AGE")
    for x in gold.get("identifiers", []):     # PHONE/EMAIL/SSN_OR_ID/USERNAME_HANDLE/OCCUPATION
        add(x["text"], x["type"])
    return dets


"""Run the full pipeline on one transcript; return everything the demos need."""
def load_case(tid: str, trace: bool = False) -> dict:
    text = (TESTS / "transcripts" / f"{tid}.txt").read_text(encoding="utf-8")
    gold = json.loads((TESTS / "gold" / f"{tid}.json").read_text(encoding="utf-8"))
    dets = resolve_overlaps(detections_from_gold(text, gold))
    mentions = make_mentions(tid, dets)
    entities, edges, info = run_pipeline(
        tid, text, mentions,
        metadata={"interview_date": gold.get("interview_date")},
        gazetteer_path=str(DATA_GAZ), trace=trace,
    )
    return {"tid": tid, "text": text, "gold": gold, "dets": dets,
            "mentions": mentions, "entities": entities, "edges": edges, "info": info}
