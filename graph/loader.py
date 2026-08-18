"""
Validates input files. Input files should be JSON files produced from the 
detection stage and formatted in the exact format shown below:

{
  "transcript_id": "transcript_014",
  "detector_version": "0.3",
  "detections": [
    {
      "start": 1042,
      "end": 1053,
      "entity_type": "PERSON",
      "score": 0.85,
      "text": "Aunt Maria",
      "recognizer": "spacy_recognizer"
    }
  ]
}
"""


from __future__ import annotations
import json
from pathlib import Path
from .models import Mention


# Identifier labels
ALLOWED_LABELS = {
    "PERSON", "NICKNAME", "LOCATION",
    "DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR",
    "AGE", "PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE",
    "DATE_OF_BIRTH", "INSTITUTION", "OCCUPATION",
}


class Violation(Exception):
    """Raised when a file is not properly formatted."""


def load_transcript(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


"""
Load detections from file and validate them. The detections file comes from the
last stage (it's supposed to be a JSON file in the format above).
"""
def load_detections(path: str | Path, transcript: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    detections = payload["detections"]

    # i = index, d = one dectection in the form {"text": "Alice", "start": 0, "end": 5...}
    for i, d in enumerate(detections):
        # check that start/end indices are correct
        actual = transcript[d["start"]:d["end"]]
        if actual != d["text"]:
            raise Violation(
                f"detection #{i}: offsets don't match text. "
                f"file says {d['text']!r}, transcript[{d['start']}:{d['end']}] "
                f"is {actual!r}"
            )
        # check that label is one of official identifiers
        if d["entity_type"] not in ALLOWED_LABELS:
            raise Violation(
                f"detection #{i}: unknown entity_type {d['entity_type']!r}"
            )
    return detections


"""
Where spans overlap, keep the longer one (ties: higher score). This is because a
longer span carries more information than a shorter one so no info is lost.
Returns a list of accepted detections.

Compared against EVERY kept span it overlaps, not just the last one. Testing
`kept[-1]` alone leaks the middle of a chain: given A=[0,10), B=[5,7), C=[6,12),
B loses to A and is replaced by... nothing (A stays), then C is compared only to A,
overlaps it, is longer, and REPLACES A -- so the 10-char span A is silently dropped in
favour of C while B's rejection stands. Any three spans where the third reaches past
the first show it. A detector that emits nested PERSON/NICKNAME spans (which this
pipeline's own upstream stage does) produces exactly that shape.
"""
def resolve_overlaps(detections: list[dict]) -> list[dict]:
    def rank(d):
        return (d["end"] - d["start"], d.get("score", 0.0))

    # Longest (then highest-scoring) first, so a span is only ever rejected by one that
    # already beat it -- which makes a single pass sufficient and the result independent
    # of input order.
    ordered = sorted(detections, key=lambda d: (-rank(d)[0], -rank(d)[1], d["start"]))
    kept: list[dict] = []
    for d in ordered:
        if any(d["start"] < k["end"] and k["start"] < d["end"] for k in kept):
            continue                                   # a better span already covers it
        kept.append(d)
    return sorted(kept, key=lambda d: d["start"])


"""
Wrap cleaned detections in Mention objects and give them stable IDs.
Example generated IDs: interview_001_m0001, interview_001_m0002... 
IMPORTANT: Each detection becomes a Mention. 
"""
def make_mentions(transcript_id: str, detections: list[dict]) -> list[Mention]:
    mentions = []
    # sort by start: start = 10, start = 50...
    # n is index, d is detection
    for n, d in enumerate(sorted(detections, key=lambda d: d["start"])):
        mentions.append(
            Mention(
                transcript_id=transcript_id,
                start=d["start"],
                end=d["end"],
                text=d["text"],
                entity_type=d["entity_type"],
                mention_id=f"{transcript_id}_m{n:04d}",  # format n to 4 digits
                score=d.get("score", 1.0),
                recognizer=d.get("recognizer", ""),
            )
        )
    return mentions


"""
Returns everything needed by the pipeline for one transcript. Assume input
directory looks like this:
data/
├── transcripts/
│   └── interview_001.txt
├── detections/
│   └── interview_001.json
└── metadata.json
In this case, transcript_id = interview_001 and data_dir = "data"
"""
def load_stage_inputs(transcript_id: str, data_dir: str | Path = "data"):
    data_dir = Path(data_dir)
    transcript = load_transcript(data_dir / "transcripts" / f"{transcript_id}.txt")
    detections = load_detections(
        data_dir / "detections" / f"{transcript_id}.json", transcript
    )
    detections = resolve_overlaps(detections)
    mentions = make_mentions(transcript_id, detections)

    meta_path = data_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text()).get(transcript_id, {})
    return transcript, mentions, metadata
