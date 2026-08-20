"""
The demo's front door: gold annotations -> perfect detections -> the real pipeline.

PURPOSE
    Give the HTML reports one function, `load_case`, that returns everything they
    need to render a walkthrough of one sample transcript.

FIT
    Sits between `samples/` and `demo/render/`. Called by
    `scripts/pipeline_report.py` and `scripts/llm_report.py`; every `stage_*`
    renderer takes the dict it returns. `evaluation/detections.py` is the
    deliberately parallel implementation for the scoring harness.

HOW
    Same trick as the evaluation harness: rather than running a real detector, the
    gold surface forms are located in the transcript and emitted as detection
    spans. The reports therefore show what THIS stage does, with detection error
    held at zero. `trace=True` is what makes the before/after clustering panels
    possible.
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

SAMPLES = REPO_ROOT / "samples"
DATA_GAZ = REPO_ROOT / "data" / "gazetteer.csv"


def all_tids():
    """Every sample transcript id, sorted -- e.g. ["interview_001", ...]."""
    # every transcript present, count-agnostic (was hardcoded interview_001..005)
    return sorted(p.stem for p in (SAMPLES / "transcripts").glob("*.txt"))


def detections_from_gold(text: str, gold: dict):
    """Turn gold surface forms into detection spans -- a stand-in perfect detector.

    Walks each gold section (people, the speaker's own name, locations, dates,
    ages, identifiers) and emits one detection per occurrence of each surface form.
    The lookarounds in the local `add` helper require a non-letter on each side, so
    "Ruth" does not match inside "Ruthie".

    Overlapping spans are fine here -- the caller runs `resolve_overlaps` next.
    """
    dets = []

    def add(surface, etype):
        """Emit one detection per occurrence of `surface`, typed `etype`."""
        for m in re.finditer(r"(?<![A-Za-z])" + re.escape(surface) + r"(?![A-Za-z])", text):
            dets.append({"text": surface, "start": m.start(), "end": m.end(),
                         "entity_type": etype, "score": 1.0})

    for p in gold["people"]:
        for f in p["forms"]:
            add(f, "PERSON")
    # The SPEAKER'S OWN name, when the transcript states it. It lives under
    # `interviewee` rather than in `people` on purpose: the identification stage folds
    # it into e000, so it is not a third party and must not be scored as one -- but a
    # detector would obviously find it, so the simulation has to emit it.
    for f in (gold.get("interviewee") or {}).get("forms", []):
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


def load_case(tid: str, trace: bool = False) -> dict:
    """Run the full pipeline on one transcript and return everything the demos need.

    Returns a "case" dict: the transcript `text`, the `gold` annotations, the
    simulated `dets` and the `mentions` built from them, plus the pipeline's
    `entities`, `edges` and `info`. Every `stage_*` renderer in `demo/render/`
    takes exactly this dict.

    `trace=True` asks the pipeline for its before/after clustering snapshots, which
    is what lets the report show the coref stage's effect separately. Note no `llm=`
    argument is passed, so this is the rules-only path; `scripts/llm_report.py`
    supplies a client by setting `KG_USE_LLM=1` in the environment.
    """
    text = (SAMPLES / "transcripts" / f"{tid}.txt").read_text(encoding="utf-8")
    gold = json.loads((SAMPLES / "gold" / f"{tid}.json").read_text(encoding="utf-8"))
    dets = resolve_overlaps(detections_from_gold(text, gold))
    mentions = make_mentions(tid, dets)
    entities, edges, info = run_pipeline(
        tid, text, mentions,
        metadata={"interview_date": gold.get("interview_date")},
        gazetteer_path=str(DATA_GAZ), trace=trace,
    )
    return {"tid": tid, "text": text, "gold": gold, "dets": dets,
            "mentions": mentions, "entities": entities, "edges": edges, "info": info}
