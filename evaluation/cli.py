"""
The console entry point: score every sample transcript and print the results.

PURPOSE
    `main` is the whole module: walk the sample transcripts, score each, print a
    per-transcript block, and finish with the aggregate.

FIT
    The top of `evaluation/`, reached via `scripts/eval.py` or
    `python3 -m evaluation.cli`. Delegates scoring to `scoring.evaluate_one` and
    all printing to `report.py`.
"""

from __future__ import annotations

from .config import ROOT
from .report import _accumulate, _print_aggregate, _print_one
from .scoring import evaluate_one


def main():
    """Score every transcript in `samples/transcripts/` and print the report.

    Transcripts are processed in sorted order so runs are comparable, and `agg`
    accumulates each result so the totals at the end cover the whole sample set.
    """
    agg = {}
    print("=" * 74)
    tids = sorted(p.stem for p in (ROOT / "transcripts").glob("*.txt"))
    for tid in tids:
        R = evaluate_one(tid)
        _print_one(R)
        _accumulate(agg, R)
    _print_aggregate(agg, len(tids))


if __name__ == "__main__":
    main()
