"""
Kin-word synonyms -> a canonical family term.

PURPOSE
    Scoring a relation's `detail` has to treat "grandma" and "grandmother" as the
    same answer, so both sides of the comparison are folded through this table.

FIT
    Used only by `evaluation/scoring.py`. A deliberate parallel to
    `graph/checks/comparators._KIN_CANON`, kept separate so the harness never
    grades the pipeline using the pipeline's own notion of agreement.

HOW
    The literal below maps one canonical term to a space-separated list of its
    variants; the loop inverts that into a variant -> canonical lookup, which is
    the direction `_canon_detail` needs.
"""

from __future__ import annotations


# kin-word synonyms -> canonical family term, for scoring relation "detail"
_KIN_CANON = {}


for _base, _variants in {
    "mother": "mother mom mommy mum mummy mama mamma momma ma",
    "father": "father dad daddy papa poppa pop pops pa",
    # `mamaw` and friends: absent here, the pipeline's correct "mamaw" edge for
    # interview_002's grandmother scored as a WRONG DETAIL against gold's
    # "grandmother" -- the same table gap that kept `graph/rules/kinship.py` from
    # producing the edge in the first place, reappearing on the scoring side.
    "grandmother": ("grandmother grandma grandmom granny nana nanna gramma grammy "
                    "meemaw mamaw mawmaw memaw mammaw mimi"),
    "grandfather": "grandfather grandpa granddad grandad grandpop gramps papaw pawpaw pappy",
    "sister": "sister sis",
    "brother": "brother bro",
}.items():
    for _v in _variants.split():
        _KIN_CANON[_v] = _base


def _canon_detail(detail: str) -> str:
    """The canonical kin term for a relation detail; unknown words pass through.

    Passing unknown words through unchanged means two identical unlisted terms
    still compare equal, so an unmapped word costs a false mismatch only against
    a DIFFERENT spelling of itself.
    """
    d = (detail or "").strip().lower()
    return _KIN_CANON.get(d, d)
