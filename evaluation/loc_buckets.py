"""
Place-type words -> a canonical bucket, for scoring LOCATION `subtype`.

PURPOSE
    Scoring a place's type has to treat "gulf" and "feature" as the same answer,
    and "county" and "region" as the same answer, so both sides of the comparison
    are folded through this table before being compared.

FIT
    Used only by `evaluation/scoring.py`. A deliberate parallel to
    `graph/checks/location.LOC_CANON`, kept separate for the same reason
    `kin_synonyms.py` is kept separate from `checks/comparators._KIN_CANON`: the
    harness must not grade the pipeline using the pipeline's own notion of
    agreement. If the pipeline widens its vocabulary, that must not silently
    widen what counts as a correct answer here.

HOW
    The literal below maps one canonical bucket to a space-separated list of the
    words that mean it; the loop inverts that into a word -> bucket lookup, which
    is the direction `_canon_type` needs.

    An unknown word passes through lowercased rather than mapping to None, so two
    identical unlisted words still compare equal and an unmapped word costs a
    false mismatch only against a DIFFERENT spelling of itself.

WHY THIS MODULE EXISTS AT ALL
    The location score used to test that `entity.subtype` was TRUTHY -- "did the
    pipeline put any type here" -- and report the result as "location typing
    accuracy". It therefore counted `Washington` typed STATE (the transcript means
    D.C.), `Mingo County` typed CITY and `Guanajuato` typed CITY as correct
    answers. Comparing against the gold type is the whole point of having a gold
    type.
"""

from __future__ import annotations


# place-type word -> canonical bucket, for scoring LOCATION `subtype`
_LOC_BUCKET = {}


for _base, _variants in {
    "country": "country nation",
    "state": "state province prefecture",
    # A county IS a region for our purposes: both are sub-state areas large
    # enough that naming one rarely singles anybody out.
    "region": "region county parish district territory island area",
    "city": "city town village municipality hamlet borough",
    "neighborhood": "neighborhood neighbourhood barrio quarter ward subdivision",
    "street": "street road avenue lane boulevard highway",
    # Natural features. The pipeline emits the SPECIFIC word it inferred ("gulf",
    # "river"), gold records the family; both are correct, so they fold together.
    "feature": ("feature gulf river creek bay bayou lake mountain mountains "
                "delta valley holler hollow sea ocean coast"),
    "institution": ("institution hospital church parish school university "
                    "college company employer clinic"),
}.items():
    for _v in _variants.split():
        _LOC_BUCKET[_v] = _base


def _canon_type(value) -> str | None:
    """The canonical bucket for a place-type word; None for an empty value.

    Accepts either side of the comparison: gold stores lowercase words ("city"),
    the pipeline stores uppercase subtypes ("CITY"), and both land on the same
    bucket. None (rather than "") for an absent type, so a caller can tell
    "no type at all" apart from "a type we do not recognise".
    """
    s = str(value or "").strip().lower()
    if not s:
        return None
    return _LOC_BUCKET.get(s, s)
