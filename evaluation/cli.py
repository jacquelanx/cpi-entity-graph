"""
`python3 -m evaluation.cli` -- score every sample transcript.
"""

from __future__ import annotations

from .config import ROOT
from .report import _accumulate, _print_aggregate, _print_one
from .scoring import evaluate_one


def main():
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
