"""
Rendering a second-line decision: action badge, checks, provenance table, flags.

PURPOSE
    Show WHY a value is what it is. Every arbitrated field carries a `Resolution`
    (see `graph/second_line/outcomes.py`), and this module turns one into readable
    HTML: what outcome it reached, which deterministic checks examined it, and what
    a reviewer still needs to look at.

FIT
    Sits directly on `primitives.py` and is used by every stage panel that displays
    an arbitrated field. Reads `graph.second_line.POLICIES` in `_flag_items`, for
    the list of real field names. Renders records; never re-derives them.

HOW
    Two presentations of the SAME record -- `_prov_table` for full-width sections
    and `_prov_list` for narrow cards -- both driven by `_action_badge` and
    `_checks_html`. The distinction `_checks_html` draws between passed, failed and
    SKIPPED checks is the point of the whole module: only a passed check means
    anything was verified.
"""

from __future__ import annotations

import re
from html import escape
from .primitives import _val


# ---------------------------------------------------------------- provenance
# The second line records, per field, HOW the value was reached. Everything below
# renders that record; nothing re-derives it.

_ACTION = {
    "confirm": ("rule + LLM agree", "a-ok"),
    "fill":    ("LLM filled the gap", "a-fill"),
    "keep":    ("rules only", "a-keep"),
    "conflict": ("layers disagreed", "a-conf"),
}


def _action_badge(res) -> str:
    """The coloured outcome label for one resolution, plus a BLOCKING mark if set.

    `reject` is split into two visually distinct labels, because the causes are
    different: "proposal refuted" means a checker did its job, while "both layers
    blind" means neither the rule nor the model produced anything -- a gap, not a
    success.
    """
    if res.action == "reject":
        label, cls = (("proposal refuted", "a-rej") if res.checks_failed
                      else ("both layers blind", "a-blind"))
    else:
        label, cls = _ACTION.get(res.action, (res.action, ""))
    blk = " <b class='blkmark'>BLOCKING</b>" if res.blocking else ""
    return f"<span class='act {cls}'>{label}</span>{blk}"


def _checks_html(res, quiet=False) -> str:
    """What actually examined this value. `checks_passed` is the only one that means
    verification -- a skipped checker said nothing, which is why they are rendered
    differently rather than lumped together.

    `quiet` is for a value on a field's SAFE direction (replace=True). Those fields
    deliberately gate only the leak-prone direction, so "nothing verified this" is
    the designed outcome rather than a gap, and flagging it amber on every redacted
    span buries the one row where it does matter -- an unverified KEEP.
    """
    bits = []
    if res.checks_passed:
        bits.append(f"<span class='ck ok'>&check; {escape(', '.join(res.checks_passed))}</span>")
    if res.checks_failed:
        bits.append(f"<span class='ck bad'>&times; {escape(', '.join(res.checks_failed))}</span>")
    if res.checks_skipped:
        bits.append(f"<span class='ck na'>n/a {escape(', '.join(res.checks_skipped))}</span>")
    if not res.checks_passed and not res.checks_failed:
        bits.insert(0, "<span class='ck na'>safe direction &mdash; no keep-gate "
                       "applies</span>" if quiet else
                       "<span class='ck none'>nothing verified this</span>")
    return " ".join(bits)


def _prov_of(e) -> dict:
    """An entity's provenance dict, or `{}` -- tolerant of an entity without one."""
    return getattr(e, "provenance", None) or {}


def _prov_table(prov: dict, names=None) -> str:
    """The decision record as a five-column table: field, outcome, value, source, checks.

    Used in the full-width sections, where columns help compare fields against each
    other. Pair-shaped fields are keyed `relation:<other_id>` /
    `same_person:<other_id>`; passing `names` renders those as
    "relation -> Aunt Maria" instead of showing a raw entity id.
    """
    rows = []
    for fname, res in prov.items():
        label = fname
        if ":" in fname and names is not None:
            base, other = fname.split(":", 1)
            label = f"{base} &rarr; {escape(names.get(other, other))}"
        else:
            label = escape(fname)
        rows.append(
            f"<tr class='{'blk' if res.blocking else ''}'>"
            f"<td class='f'>{label}</td>"
            f"<td>{_action_badge(res)}</td>"
            f"<td class='v'>{_val(res.value)}</td>"
            f"<td class='src2'>{escape(res.source or '&mdash;')}"
            + (f"<i> &middot; {escape(res.confidence)} conf</i>"
               if res.confidence and res.confidence != "unstated" else "")
            + f"</td><td class='ck-cell'>{_checks_html(res)}"
              f"<div class='why'>{escape(res.reason)}</div></td></tr>")
    if not rows:
        return "<p class='muted'>No fields were resolved for this entity.</p>"
    return ("<table class='prov'><thead><tr><th>field</th><th>outcome</th>"
            "<th>value</th><th>source</th><th>deterministic checks</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def _prov_list(prov: dict, names=None) -> str:
    """The same decision record as `_prov_table`, stacked instead of tabulated.

    A five-column table needs ~690px; a person card in the grid is ~300px, so the
    table version overflowed its card and rendered as an unreadable smear. Inside a
    card each field becomes a small block that wraps cleanly at any width; the table
    is kept for the full-width sections, where columns genuinely help comparison.
    """
    blocks = []
    for fname, res in prov.items():
        if ":" in fname and names is not None:
            base, other = fname.split(":", 1)
            label = f"{escape(base)} &rarr; {escape(names.get(other, other))}"
        else:
            label = escape(fname)
        src = escape(res.source or "&mdash;")
        if res.confidence and res.confidence != "unstated":
            src += f", {escape(res.confidence)} confidence"
        blocks.append(
            f"<div class='pv{' blk' if res.blocking else ''}'>"
            f"<div class='pv-h'><span class='pv-f'>{label}</span>"
            f"{_action_badge(res)}</div>"
            f"<div class='pv-v'>{_val(res.value)} <i>&middot; {src}</i></div>"
            f"<div class='pv-c'>{_checks_html(res)}</div>"
            f"<div class='why'>{escape(res.reason)}</div></div>")
    return f"<div class='pv-list'>{''.join(blocks)}</div>"


def _prov_details(e, names=None, label="decision record", stacked=False) -> str:
    """The whole record wrapped in a collapsed `<details>`, with a blocking count.

    Collapsed by default so a page with fifty entities stays readable, and the
    summary line carries the field count plus any blocking total -- enough to
    decide whether to open it. `stacked=True` picks the narrow card layout.
    Returns "" for an entity with no provenance at all.
    """
    prov = _prov_of(e)
    if not prov:
        return ""
    n_blk = sum(1 for r in prov.values() if r.blocking)
    tail = f" &middot; <b class='blkmark'>{n_blk} blocking</b>" if n_blk else ""
    inner = _prov_list(prov, names) if stacked else _prov_table(prov, names)
    return (f"<details class='prov-d'><summary>{label} &middot; {len(prov)} field(s)"
            f"{tail}</summary>{inner}</details>")


# Review flags are accumulated by `Entity.flag_entity`, which joins them with "; ".
# Rendered raw that is one long red run-on sentence -- and the reasons THEMSELVES
# contain semicolons and colons, so a naive split shreds them. Each flag written by
# `apply_resolution` starts with "<field>: ", so split only where the next fragment
# begins with a real policy field name.
def _flag_items(reason: str) -> list[str]:
    """Split an accumulated `review_reason` string back into individual flags.

    `Entity.flag_entity` joins reasons with "; ", but the reasons THEMSELVES
    contain semicolons and colons, so splitting on the separator shreds them. The
    fix: split only where the text after the separator begins with a real policy
    FIELD NAME followed by a colon -- because every flag `apply_resolution` writes
    starts with "<field>: ".

    Field names are sorted longest-first so the alternation prefers the most
    specific match (e.g. `replace_location` over `replace`).
    """
    from graph.second_line import POLICIES
    fields = sorted(set(POLICIES) | {"same_person"}, key=len, reverse=True)
    pat = "|".join(re.escape(f) for f in fields)
    return [p.strip() for p in re.split(rf";\s+(?={pat}:)", reason or "") if p.strip()]


def _flag_html(e) -> str:
    """The review flags for one entity, as a readable list rather than a red blob."""
    if not e.needs_review or not e.review_reason:
        return ""
    items = []
    for item in _flag_items(e.review_reason):
        head, sep, rest = item.partition(": ")
        if sep and " " not in head:                # "<field>: <why>"
            items.append(f"<li><b>{escape(head)}</b> {escape(rest)}</li>")
        else:
            items.append(f"<li>{escape(item)}</li>")
    return (f"<div class='note'><div class='note-h'>&#9873; needs review "
            f"&middot; {len(items)}</div><ul>{''.join(items)}</ul></div>")
