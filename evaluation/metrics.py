"""
Two primitives shared by scoring and reporting.

PURPOSE
    An accuracy ratio, and the stand-in object used when a gold row has no
    matching entity.

FIT
    They live here so `scoring.py` and `report.py` do not have to import each
    other -- both need them, neither should depend on the other.
"""

from __future__ import annotations


def _acc(num, den):
    """`num / den`, or None when there is nothing to measure.

    None rather than 0.0 for an empty denominator, because "no cases" is not the
    same as "got them all wrong" -- and `report._fmt` renders None as "n/a".
    """
    return (num / den) if den else None


class _NoEnt:
    """Stand-in so a missing entity reads as 'untyped' instead of raising."""
    subtype = None
