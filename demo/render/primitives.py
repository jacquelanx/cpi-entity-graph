"""
Small shared formatting helpers and the palette/label tables.

PURPOSE
    The vocabulary the stage renderers are written in: category colours, display
    labels, and a handful of one-line formatters.

FIT
    The bottom of `demo/render/` -- everything here is pure string formatting with
    no knowledge of a pipeline stage, and every other module in the package builds
    on it. The only project import is `graph.models.Relation`, for rendering an
    edge label.

HOW
    `_HL` maps each entity category to a `(background, text)` colour pair, so the
    same category is the same colour everywhere in the report. The rest are small
    escaping / formatting helpers -- note that all of them go through
    `html.escape`, since transcript text is untrusted input as far as HTML is
    concerned.
"""

from __future__ import annotations

from html import escape
from graph.models import Relation


# ---- category hues (fixed; readable on light surfaces) ----
_HL = {
    "PERSON": ("#E9F0FA", "#234E86"), "LOCATION": ("#EAF2E0", "#315915"),
    "INSTITUTION": ("#EAF2E0", "#315915"), "AGE": ("#EFEDFA", "#413593"),
    "DATE_ABSOLUTE": ("#F8EFDD", "#6A4310"), "DATE_RELATIVE": ("#F8EFDD", "#6A4310"),
    "DATE_ANCHOR": ("#FBEAEA", "#7C2222"), "DATE_OF_BIRTH": ("#F8EFDD", "#6A4310"),
    "PHONE": ("#FBEAEA", "#7C2222"), "EMAIL": ("#FBEAEA", "#7C2222"),
    "SSN_OR_ID": ("#FBEAEA", "#7C2222"), "USERNAME_HANDLE": ("#FBEAEA", "#7C2222"),
    "OCCUPATION": ("#E4F1F1", "#245b5b"),
}


_PERSON_FILL = {"FAMILY": "#5B7FA6", "PROFESSIONAL": "#9CB6D2",
                "PUBLIC_FIGURE": "#AEB3BA", "PUBLIC_FIGURE_UNCONFIRMED": "#AEB3BA"}


_ID_CATS = ("PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "OCCUPATION")


_DATE_CATS = ("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR", "DATE_OF_BIRTH")


_ID_LABEL = {"PHONE": "Phone", "EMAIL": "Email", "SSN_OR_ID": "SSN / ID",
             "USERNAME_HANDLE": "Handle", "OCCUPATION": "Occupation"}


def _relname(ed):
    """An edge's relation as a plain string, whether it is an enum or already a str."""
    return ed.relation.value if isinstance(ed.relation, Relation) else str(ed.relation)


def _pname(e):
    """An entity's display name: its longest surface form, or "?" if it has none.

    `sorted_mentions` is longest-first, so element 0 is the most complete form the
    transcript used ("Aunt Maria" rather than "Maria").
    """
    return e.sorted_mentions[0] if e.sorted_mentions else "?"


def _pct(x):
    """A 0..1 ratio as a whole-number percentage; None renders as an em dash."""
    return "&mdash;" if x is None else f"{x * 100:.0f}%"


def _val(v):
    """Render any attribute value as display HTML.

    None becomes a greyed em dash (meaning "not established", which is visually
    distinct from a value of "no"), booleans become yes/no, and everything else is
    escaped -- values can contain transcript text, so escaping is required.
    """
    if v is None:
        return "<i class='none'>&mdash;</i>"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return escape(str(v))


def section(num, title, body, note="", wide=False):   # `wide` kept for callers; ignored
    """Wrap a stage's body HTML in the numbered `<section>` the page expects.

    `num` is the stage number shown in the stepper, `note` an optional subtitle.
    """
    note_html = f"<p class='s-note'>{note}</p>" if note else ""
    return (f"<section class='stage'><div class='s-head'><span class='s-num'>{num}</span>"
            f"<h3>{escape(title)}</h3></div>{note_html}{body}</section>")


def _chip(text, cls=""):
    """A small inline label. `cls` adds a variant class (e.g. a badge colour).

    NOTE: `text` is inserted RAW, so callers pass either already-escaped text or
    deliberate markup.
    """
    return f"<span class='{('chip ' + cls).strip()}'>{text}</span>"


def _names_map(case) -> dict:
    """`{entity_id: display name}` for every entity in a case.

    The interviewee is overridden last: an unnamed speaker reads as "the
    interviewee" rather than as "?" or a raw id, which is what a person looking at
    the report needs to see.
    """
    names = {e.entity_id: _pname(e) for e in case["entities"]}
    iv = case["info"]["interviewee"]
    names[iv.entity_id] = _pname(iv) if iv.sorted_mentions else "the interviewee"
    return names
