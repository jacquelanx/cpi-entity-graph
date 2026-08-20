"""
Input boundary: read and validate the upstream detection stage's output.

PURPOSE
    Turns the two files that describe one interview -- the raw `.txt` transcript
    and the detector's `.json` span list -- into the exact objects the pipeline
    wants: a transcript string plus a clean, non-overlapping, stably-numbered
    list of `Mention`s. Rejects malformed input loudly rather than letting a bad
    offset propagate.

FIT
    First stage in the flow, and the only place that knows the detector's file
    format. `scripts/build_graph.py` and `demo/cases.py` call it (or hand-build
    equivalent mentions) and pass the result to `graph/pipeline.run_pipeline`.
    Depends only on `graph/models.Mention`; nothing in `graph/` depends on this
    module except the entry-point scripts, which keeps the format contract in one
    file.

HOW
    Three steps, deliberately separate so each can be tested and reused alone:

      1. `load_detections` validates -- every span must slice back to its own
         text, and every label must be one of the fourteen agreed categories.
      2. `resolve_overlaps` reduces overlapping spans to a non-overlapping set,
         preferring the longest span (see that function for why order matters).
      3. `make_mentions` assigns transcript-order ids (`interview_001_m0000`,
         `_m0001`, ...) so the same input always produces the same ids -- which
         is what lets reports, tests and the artifact refer to a mention by id.

INPUT FORMAT
    Detections files are JSON, exactly this shape:

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

    `start`/`end` are character offsets into the raw `.txt`, 0-indexed and
    end-exclusive. `score` and `recognizer` are optional. Overlapping spans are
    allowed here and resolved below.
"""


from __future__ import annotations
import json
from pathlib import Path
from .models import Mention


# The closed set of entity types the detection stage is allowed to emit. Anything
# else is a contract violation rather than something to silently drop, because a
# label this stage does not recognize would flow through unhandled to surrogate
# generation and leak whatever it marked.
ALLOWED_LABELS = {
    "PERSON", "NICKNAME", "LOCATION",
    "DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR",
    "AGE", "PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE",
    "DATE_OF_BIRTH", "INSTITUTION", "OCCUPATION",
}


class Violation(Exception):
    """Raised when a file is not properly formatted."""


def load_transcript(path: str | Path) -> str:
    """Read a transcript file as UTF-8 text.

    Encoding is pinned rather than left to the platform default: every offset in
    every detections file counts characters in THIS decoding, so decoding the
    same bytes differently on another machine would shift every span.
    """
    return Path(path).read_text(encoding="utf-8")


def load_detections(path: str | Path, transcript: str) -> list[dict]:
    """Load a detections JSON file and reject it if it disagrees with the transcript.

    Two invariants are checked per detection, both fatal:

      * OFFSET AGREEMENT -- `transcript[start:end]` must equal `text` exactly.
        This is the load-bearing one. Surrogate generation later splices
        replacement text at these offsets, so a file whose spans were computed
        against a different revision of the transcript would silently redact the
        wrong characters. Comparing the slice to the text catches that here, and
        the error message prints both strings so the mismatch is obvious.
      * KNOWN LABEL -- `entity_type` must be in `ALLOWED_LABELS`.

    Returns the raw detection dicts unchanged (still possibly overlapping); use
    `resolve_overlaps` next.
    """
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


def resolve_overlaps(detections: list[dict]) -> list[dict]:
    """Reduce overlapping spans to a non-overlapping set, keeping the longest.

    WHY: detectors routinely emit nested spans for the same text -- "Aunt Maria"
    as PERSON and "Maria" as PERSON, or a NICKNAME inside a PERSON. Downstream
    stages assume one mention per stretch of text, and the longer span carries
    strictly more information, so no detail is lost by preferring it. Ties break
    on `score`, then on `start` for determinism.

    HOW: sort every candidate best-first (longest, then highest-scoring), then
    walk the sorted list keeping a span only if it overlaps NOTHING already kept.
    Sorting first is what makes one pass correct: by the time a span is
    considered, anything that could beat it has already been decided, so a
    rejection is final and the result does not depend on input order.

    The `any(...)` compares against EVERY kept span, not just the last one.
    Testing `kept[-1]` alone leaks the middle of a chain: given A=[0,10),
    B=[5,7), C=[6,12), B loses to A and is replaced by... nothing (A stays), then
    C is compared only to A, overlaps it, is longer, and REPLACES A -- so the
    10-char span A is silently dropped in favour of C while B's rejection stands.
    Any three spans where the third reaches past the first show it. A detector
    that emits nested PERSON/NICKNAME spans (which this pipeline's own upstream
    stage does) produces exactly that shape.

    Two half-open intervals overlap exactly when each starts before the other
    ends -- `d.start < k.end and k.start < d.end` -- which is the test used here.
    Adjacent spans like [0,5) and [5,9) therefore do NOT count as overlapping.

    Returns the accepted detections back in transcript order.
    """
    def rank(d):
        """A span's quality as `(length, score)` -- longer wins, then higher-scoring."""
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


def make_mentions(transcript_id: str, detections: list[dict]) -> list[Mention]:
    """Wrap cleaned detections in `Mention` objects with stable, ordered ids.

    Ids are assigned in TRANSCRIPT order (`interview_001_m0000`, `_m0001`, ...),
    not in whatever order the detector emitted, so the same transcript plus the
    same detections always yields the same id for the same span. Reports, tests
    and the saved artifact all cite mentions by id, so that stability is what
    makes them comparable across runs.

    `f"{n:04d}"` zero-pads the counter to four digits, which keeps ids the same
    width and therefore sortable as plain strings ("m0009" < "m0010", whereas
    "m9" > "m10").
    """
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


def load_stage_inputs(transcript_id: str, data_dir: str | Path = "data"):
    """Load, validate and prepare everything the pipeline needs for one transcript.

    This is the convenience front door: it runs the whole input boundary in order
    (read text -> validate detections -> resolve overlaps -> make mentions) and
    returns `(transcript, mentions, metadata)` ready to hand to
    `pipeline.run_pipeline`.

    Expects this layout, with `transcript_id="interview_001"` and
    `data_dir="data"`:

        data/
        |-- transcripts/
        |   `-- interview_001.txt
        |-- detections/
        |   `-- interview_001.json
        `-- metadata.json

    `metadata.json` is optional and holds a dict keyed by transcript id (its
    `interview_date` anchors relative dates like "two years ago"). A missing file
    or a missing key yields `{}` rather than an error, so a transcript with no
    known interview date still runs -- relative-date resolution just abstains.
    """
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
