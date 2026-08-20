"""
Rendering a second-line decision: the action badge, the list of
deterministic checks that ran, the provenance table, and the review flags.

This is the part of the report that shows WHY a value is what it is, so it is
shared by every stage panel that displays an arbitrated field.
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
    return getattr(e, "provenance", None) or {}


def _prov_table(prov: dict, names=None) -> str:
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
