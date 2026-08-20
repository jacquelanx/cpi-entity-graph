"""
The one primitive shared by scoring and reporting.

PURPOSE
    An accuracy ratio that says "not measurable" rather than "zero" when there is
    nothing to measure.

FIT
    It lives here so `scoring.py` and `report.py` do not have to import each
    other -- both need it, neither should depend on the other.

HISTORY
    This module also held `_NoEnt`, a stand-in object whose only job was to let
    `(entity_for(...) or _NoEnt).subtype` read as "untyped" instead of raising.
    That expression WAS the location-typing metric, and testing an attribute for
    truthiness is not the same as comparing it to gold -- see the note at the top
    of `loc_buckets.py`. With the comparison written properly the stand-in has no
    caller.
"""

from __future__ import annotations


def _acc(num, den):
    """`num / den`, or None when there is nothing to measure.

    None rather than 0.0 for an empty denominator, because "no cases" is not the
    same as "got them all wrong" -- and `report._fmt` renders None as "n/a".
    """
    return (num / den) if den else None
