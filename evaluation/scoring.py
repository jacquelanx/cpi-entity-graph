"""
Scoring one transcript against its gold annotations.

PURPOSE
    `evaluate_one` is the measurement: run the real pipeline on simulated-perfect
    detections, then compare what came out against the hand-written gold file
    field by field. Returns one result dict per transcript.

FIT
    The heart of `evaluation/`. Called by `cli.py`, formatted by `report.py`.
    Imports `graph.pipeline.run_pipeline` -- the SAME entry point the demos and
    reports use, so a value the second line confirmed, filled or rejected is
    scored exactly as a consumer would see it. `config` must be imported before
    `graph.pipeline` (see that module).

HOW -- four ideas worth knowing before reading
    1. GOLD IS KEYED BY SURFACE TEXT, with no offsets. One gold row can therefore
       correspond to SEVERAL pipeline entities -- AGE entities are one-per-mention,
       so two occurrences of "twelve" are two entities. `entities_for` returns all
       of them and the individual scorers pick the right one, rather than grading
       whichever happened to come first. Where the occurrences DISAGREE (one is a
       flood depth, one is a child's age) the gold row carries a `context` snippet
       that pins it to a single span, because "best of N" is a free pass once the
       candidates give different answers.
    2. ERRORS ARE SPLIT BY DIRECTION, not just counted. A LEAK (should have been
       redacted, was kept) is unrecoverable; an over-redaction costs only narrative
       colour. Reporting one number for both would hide the distinction that
       matters. The same reasoning splits `reject` into "refuted" and "blind" at
       the bottom of the file.
    3. A GOLD ROW WITH NO MATCHING ENTITY IS A FAILURE, not an absence. Every
       scorer here used to `continue` past those rows, which removed them from the
       numerator AND the denominator -- so the one outcome the pipeline can never
       recover from (it built nothing at all for a span gold says exists) was the
       one outcome no percentage could show. They now count, in the direction their
       field makes them dangerous: a leak for `replace`, a miss for a value, an
       unresolved owner for `owner`.
    4. EVERY BOOLEAN METRIC REPORTS ITS GOLD CLASS BALANCE (`gold_pos` /
       `gold_neg`, via `_balance`). An all-True or all-False gold set is satisfied
       by a constant answer, so its percentage says nothing about the pipeline;
       printing the balance beside it is what stops a 100% from being read as a
       result. `identifying` is the standing example -- see the note there.
"""

from __future__ import annotations

import json
from collections import Counter

# FIRST: puts the repo root on sys.path and quiets the fastcoref / transformers
# loggers. Both have to happen before `graph.pipeline` pulls those libraries in.
from .config import DATA_GAZ, ROOT, RUN_COREF, _LLM

from dateutil import parser as dateparser
from graph.loader import resolve_overlaps, make_mentions
from graph.pipeline import run_pipeline
from graph.models import Relation
from graph.rules.identifiers import ID_CATS
from .detections import _build_detections
from .kin_synonyms import _canon_detail
from .loc_buckets import _canon_type
from .metrics import _acc


def _canon_subtype(value):
    """A PERSON subtype folded to its FAMILY of answers, or None.

    `PUBLIC_FIGURE_UNCONFIRMED` is the pipeline's "on the closed list, but the LLM
    never affirmed it" state -- the same subtype with a confidence qualifier
    attached, and `checks/persons.py` itself tests it with
    `sub.startswith("PUBLIC_FIGURE")`. Grading the qualifier as a WRONG subtype
    would be the harness inventing an error, so the family is what gets compared
    and the qualifier is counted separately (`unconfirmed`) instead of hidden.
    """
    s = str(value or "").strip().upper()
    if not s:
        return None
    return "PUBLIC_FIGURE" if s.startswith("PUBLIC_FIGURE") else s


def _balance(rows, key):
    """How many gold rows assert True and how many assert False for `key`.

    Reported alongside every boolean metric so a SINGLE-CLASS gold set is visible
    in the output instead of being hidden behind a percentage. A metric whose gold
    is all-True is satisfied by a constant "True" and carries no information about
    the pipeline; `report._fmt_bal` marks those rows rather than printing an
    accuracy that cannot fall.
    """
    vals = [bool(r[key]) for r in rows if key in r]
    return {"gold_pos": sum(vals), "gold_neg": len(vals) - sum(vals)}


def _relation_report(gold_rel: dict, pred_rel: dict, pred_src: dict) -> dict:
    """Score predicted relations against gold and break the result down by which
    layer produced each edge.

    `gold_rel` and `pred_rel` both map a `(source, target)` pair to its relation
    detail. `pred_src` maps a pair to 'rule' or 'llm'.

    HOW: SET ARITHMETIC over the pair keys. The intersection is the true positives,
    `pred - gold` the false positives, `gold - pred` the misses. `detail_ok` then
    counts how many true positives also got the RELATION WORD right -- finding
    that two people are related but calling an aunt a sister is a partial credit
    case, so it is measured separately from recall.

    The same arithmetic is then repeated per LAYER. `llm_gain` -- the gold
    relations the LLM found that the rules missed -- is the recall the LLM path
    actually buys, and `llm_fp` is what it costs.
    """
    gold_set, pred_set = set(gold_rel), set(pred_rel)
    tp = gold_set & pred_set
    detail_ok = sum(1 for k in tp if gold_rel[k] == pred_rel[k])
    rule_pred = {k for k in pred_set if pred_src.get(k) == "rule"}
    llm_pred = {k for k in pred_set if pred_src.get(k) == "llm"}
    rule_tp, llm_tp = rule_pred & gold_set, llm_pred & gold_set
    return {
        "gold": len(gold_rel), "pred": len(pred_set), "tp": len(tp),
        "detail_ok": detail_ok,
        "precision": _acc(len(tp), len(pred_set)),
        "recall": _acc(len(tp), len(gold_rel)),
        "detail_acc": _acc(detail_ok, len(tp)),
        "misses": sorted(gold_set - pred_set),
        "false_pos": sorted(pred_set - gold_set),
        "bad_detail": sorted(k for k in tp if gold_rel[k] != pred_rel[k]),
        # provenance breakdown
        "rule_pred": len(rule_pred), "llm_pred": len(llm_pred),
        "rule_tp": len(rule_tp), "llm_tp": len(llm_tp),
        "rule_fp": len(rule_pred - gold_set), "llm_fp": len(llm_pred - gold_set),
        "llm_gain": sorted(llm_tp - rule_tp),        # gold rels the LLM added over rules
        "llm_false_pos": sorted(llm_pred - gold_set),
    }


def evaluate_one(tid: str) -> dict:
    """Score one transcript end to end and return every metric for it.

    Steps: read the transcript and its gold file -> build simulated-perfect
    detections -> run the real pipeline (with `trace=True`, so clustering
    before/after is available) -> score each field group in turn.

    The result dict has one key per scored area -- `cluster`, `rel`, `alias`,
    `gender`, `person_subtype`, `role`, `given_name`, `surname`, `ethnicity`,
    `replace`, `dates`, `ages`, `shiftable`, `approximate`, `stated_with`, `kind`,
    `locations`, `loc_parent`, `owner`, the two redaction groups, `identifying`,
    `interviewee` and `second_line` -- each holding its own counts and ratios.
    `report.py` knows that shape.

    That list is now every field in `graph/second_line/policies.POLICIES`. Two are
    scored indirectly and deliberately: `same_person` through `alias` (positive
    direction) plus `cluster.over_merges` (negative), and `resolved_value` /
    `value` through the `dates` / `ages` blocks.

    The nested helpers exist because they all close over `entities` / `gold` for
    this one transcript: `ents_of` filters by category, `entities_for` /
    `entity_for` map a gold surface form back to entities, and `canon_of_entity`
    maps a clustered entity onto the gold canonical name(s) its mentions belong
    to -- which is how clustering itself is scored.
    """
    text = (ROOT / "transcripts" / f"{tid}.txt").read_text(encoding="utf-8")
    gold = json.loads((ROOT / "gold" / f"{tid}.json").read_text(encoding="utf-8"))

    # ---- run THE REAL pipeline on simulated-perfect detections ----
    dets = resolve_overlaps(_build_detections(text, gold))
    mentions = make_mentions(tid, dets)

    entities, edges, info = run_pipeline(
        tid, text, mentions,
        metadata={"interview_date": gold.get("interview_date")},
        gazetteer_path=str(DATA_GAZ), run_coref=RUN_COREF, trace=True, llm=_LLM)

    interviewee = info["interviewee"]
    ledger = info.get("ledger", {})
    coref_merges = len(info.get("coref_merges", []))

    persons = [e for e in entities if e.category == "PERSON" and e is not interviewee]
    by_id = {e.entity_id: e for e in entities}

    def ents_of(*categories):
        """Every entity in one of the given categories."""
        return [e for e in entities if e.category in categories]

    def entities_for(text_value: str, pool, context: str | None = None):
        """EVERY pipeline entity whose mentions include this gold surface form.

        Gold is keyed by surface TEXT with no offsets, so one gold row can legitimately
        correspond to several entities: AGE entities are one-per-mention (see
        `pipeline._simple_entities`), so "twelve" in "the water came up twelve feet"
        and "twelve" in "my daughter Trang was maybe twelve" are two entities. Picking
        the first and scoring only that graded whichever span happened to come first in
        the transcript -- and marked the age WRONG when the leading span was the
        measurement `checks/ages.not_a_measurement` had correctly refused.

        `context` PINS a text-keyed row to ONE occurrence: it is a snippet of the
        transcript, and only entities with a mention inside that snippet's span
        qualify. Without it two gold rows sharing a surface form are
        indistinguishable, which is what stopped the corpus from carrying both
        readings of "twelve" -- the flood depth that must be KEPT and the child's
        age that must be REPLACED. Choosing the best of several candidates is
        legitimate when a gold row really does describe every occurrence; it is a
        free pass when the occurrences disagree.
        """
        t = text_value.lower()
        found = [e for e in pool if any(m.text.lower() == t for m in e.mentions)]
        if context is None:
            return found
        at = text.find(context)
        if at < 0:                      # a context that does not occur is an error
            raise ValueError(f"{tid}: gold context {context!r} is not in the transcript")
        lo, hi = at, at + len(context)
        return [e for e in found
                if any(m.start >= lo and m.end <= hi for m in e.mentions)]

    def entity_for(text_value: str, pool, context: str | None = None):
        """The first such entity, for the scorers that need exactly one."""
        found = entities_for(text_value, pool, context)
        return found[0] if found else None

    # ---- surface form -> gold canonical ----
    form2canon = {}
    for p in gold["people"]:
        for form in p["forms"]:
            form2canon[form.lower()] = p["canonical"]

    def canon_of_entity(e):
        """The set of GOLD canonical names this entity's mentions belong to.

        The basis of clustering scores. Each mention is looked up in the
        surface-form -> canonical map built above; the resulting set tells us what
        the entity actually is:

          exactly one canonical  -> a clean cluster;
          two or more            -> an OVER-MERGE (two real people fused);
          empty                  -> no mention matched gold (e.g. the interviewee).

        And a canonical appearing in several entities' sets is a SPLIT.
        """
        votes = {}
        for m in e.mentions:
            c = form2canon.get(m.text.lower())
            if c:
                votes[c] = votes.get(c, 0) + 1
        if not votes:
            return set()
        return set(votes)

    ent_canon = {}      # entity_id -> majority gold canonical (for relation ends)
    ent_cset = {}       # entity_id -> set of gold canonicals it touches
    for e in persons:
        cset = canon_of_entity(e)
        ent_cset[e.entity_id] = cset
        votes = {}
        for m in e.mentions:
            c = form2canon.get(m.text.lower())
            if c:
                votes[c] = votes.get(c, 0) + 1
        ent_canon[e.entity_id] = max(votes, key=votes.get) if votes else None
    ent_canon[interviewee.entity_id] = "INTERVIEWEE"

    canon_to_entities = {}
    for e in persons:
        for c in ent_cset[e.entity_id]:
            canon_to_entities.setdefault(c, set()).add(e.entity_id)

    R = {"tid": tid}

    # ---------- CLUSTERING ----------
    gold_canons = [p["canonical"] for p in gold["people"]]
    over_merges = sum(1 for e in persons if len(ent_cset[e.entity_id]) > 1)
    splits = sum(1 for c in gold_canons if len(canon_to_entities.get(c, set())) > 1)
    exact = 0
    for c in gold_canons:
        ents = canon_to_entities.get(c, set())
        if len(ents) == 1 and ent_cset[next(iter(ents))] == {c}:
            exact += 1
    R["cluster"] = {"gold": len(gold_canons), "exact": exact,
                    "over_merges": over_merges, "splits": splits,
                    "coref_merges": coref_merges,
                    "recall": _acc(exact, len(gold_canons))}

    # ---------- RELATIONS ----------
    gold_rel = {(r["source"], r["target"]): _canon_detail(r["detail"])
                for r in gold["relations"]}
    pred_rel, pred_src = {}, {}
    for ed in edges:
        if ed.relation != Relation.RELATED_TO:
            continue
        s, t = ent_canon.get(ed.source, "?"), ent_canon.get(ed.target, "?")
        pred_rel[(s, t)] = _canon_detail(ed.detail)
        # provenance: LLM-added edges are tagged "(llm)" in their evidence
        pred_src[(s, t)] = "llm" if str(ed.evidence).startswith("(llm)") else "rule"
    R["rel"] = _relation_report(gold_rel, pred_rel, pred_src)

    # ---------- GENDER (only where gold gender known) ----------
    g_total = g_correct = g_wrong = g_missing = 0
    for p in gold["people"]:
        if p["gender"] not in ("F", "M"):
            continue
        g_total += 1
        preds = {by_id[eid].attributes.get("gender")
                 for eid in canon_to_entities.get(p["canonical"], set())}
        preds.discard(None)
        if not preds:
            g_missing += 1
        elif preds == {p["gender"]}:
            g_correct += 1
        else:
            g_wrong += 1
    R["gender"] = {"total": g_total, "correct": g_correct, "wrong": g_wrong,
                   "missing": g_missing, "recall": _acc(g_correct, g_total)}

    # ---------- PERSON ATTRIBUTES (subtype / role / name parts / ethnicity) ----------
    # One helper for every per-person field, because they all score the same shape
    # and all three outcomes matter differently: a WRONG value is actively
    # misleading downstream, a MISSING one is a gap somebody can still fill, and
    # gold `null` means "no value is the right answer" so inventing one is wrong
    # rather than free.
    #
    # A name split across entities that disagree is not a correct answer -- hence
    # the single-element check rather than "does any entity have it".
    def _person_attr(key, get, canon=None, qualified=None):
        """Three-way score of one PERSON attribute against `gold["people"][key]`.

        `get(entity)` reads the pipeline's value, `canon` folds both sides to a
        comparable form (kin synonyms for `role`, the PUBLIC_FIGURE family for
        `subtype`), and `qualified` optionally counts values the pipeline hedged.
        """
        canon = canon or (lambda v: str(v).strip().lower() if v is not None
                          and str(v).strip() else None)
        out = {"total": 0, "correct": 0, "wrong": 0, "missing": 0, "fail": []}
        if qualified is not None:
            out["qualified"] = 0
        for p in gold["people"]:
            if key not in p:
                continue
            out["total"] += 1
            want = canon(p[key])
            raw = {get(by_id[eid])
                   for eid in canon_to_entities.get(p["canonical"], set())}
            raw = {v for v in raw if v is not None and str(v).strip()}
            if qualified is not None and any(qualified(v) for v in raw):
                out["qualified"] += 1
            got = {canon(v) for v in raw}
            got.discard(None)
            shown = "/".join(sorted(str(v) for v in raw))
            one = next(iter(got)) if len(got) == 1 else None
            if want is None:
                if not got:
                    out["correct"] += 1
                else:
                    out["wrong"] += 1
                    out["fail"].append((p["canonical"], shown, None))
            elif one == want:
                out["correct"] += 1
            elif not got:
                out["missing"] += 1
                out["fail"].append((p["canonical"], None, p[key]))
            else:
                out["wrong"] += 1
                out["fail"].append((p["canonical"], shown, p[key]))
        out["accuracy"] = _acc(out["correct"], out["total"])
        return out

    # ---------- PERSON SUBTYPE ----------
    # FAMILY / PROFESSIONAL / PUBLIC_FIGURE / none. Gold has carried this field
    # since the first sample set and NOTHING read it, so the pipeline's answers
    # here were never graded -- including interview_002's mine foreman, who is
    # typed FAMILY because the containment over-merge fused him with the speaker's
    # uncle. Downstream that hands a stranger a relative's surrogate identity.
    #
    # Scored three ways for the same reason ownership is: a WRONG subtype is
    # actively misleading, a MISSING one is a gap somebody can still fill. Gold
    # `null` means "no subtype is the correct answer", so inventing one there is a
    # wrong answer, not a bonus.
    R["person_subtype"] = _person_attr(
        "subtype", lambda e: e.subtype, canon=_canon_subtype,
        qualified=lambda v: str(v).upper().endswith("_UNCONFIRMED"))
    R["person_subtype"]["unconfirmed"] = R["person_subtype"].pop("qualified", 0)

    # ---------- ROLE / NAME PARTS / ETHNICITY ----------
    # Three fields the arbitration decides on every person and nothing graded.
    #
    #   role       what surrogate generation calls this person. Folded through the
    #              kin synonyms, so the pipeline's "mamaw" matches gold's
    #              "grandmother" -- the same fold the relation `detail` gets, for
    #              the same reason.
    #   given_name / surname  the two halves a surrogate NAME is minted from. Which
    #              half a token lands in is the whole decision: slotting a surname
    #              as a given name mints a fake first name to stand in for a family
    #              name, and the honorific-vs-kin-prefix split in
    #              `name_matching.split_name_parts` is what gets it right or wrong.
    #   ethnicity  scored for THIRD PARTIES here; the speaker's own is in the
    #              interviewee block below. Gold `null` where the transcript
    #              attributes none, so an invented ethnicity is a wrong answer --
    #              which is what the refutation checkers exist to prevent.
    R["role"] = _person_attr("role", lambda e: e.attributes.get("role"),
                             canon=lambda v: _canon_detail(v) if v else None)
    R["given_name"] = _person_attr("given_name",
                                   lambda e: e.attributes.get("given_name"))
    R["surname"] = _person_attr("surname", lambda e: e.attributes.get("surname"))
    R["ethnicity"] = _person_attr("ethnicity",
                                  lambda e: e.attributes.get("ethnicity"))

    # ---------- ALIAS MERGE (the `same_person` positive direction) ----------
    # Clustering recall already punishes a split, but it mixes the alias path in
    # with everything else. Every gold person with more than one surface form is an
    # alias case -- "we called her Sissy", "everybody called her Pera" -- and this
    # isolates them, so a regression in `rules/aliases.py` or the coref gate shows
    # up as its own number instead of moving one point of clustering recall.
    # The negative direction is `cluster.over_merges`: two gold canonicals sharing
    # one entity is exactly a wrong `same_person`.
    al_rows = [p for p in gold["people"] if len(p["forms"]) > 1]
    al_ok, al_fail = 0, []
    for p in al_rows:
        ents = canon_to_entities.get(p["canonical"], set())
        if len(ents) == 1:
            al_ok += 1
        else:
            al_fail.append((p["canonical"], p["forms"]))
    R["alias"] = {"total": len(al_rows), "merged": al_ok,
                  "accuracy": _acc(al_ok, len(al_rows)), "fail": al_fail}

    # ---------- REPLACE / SAFETY ----------
    # A gold person with NO entity used to default to `pred_replace = True`, i.e. a
    # person the pipeline never built at all scored as a CORRECT redaction. Nothing
    # downstream redacts a span no entity covers, so that is a leak -- count it as
    # one. (`missing` is reported separately so the two causes stay distinguishable.)
    rp_total = rp_correct = leaks = over_red = rp_missing = 0
    for p in gold["people"]:
        rp_total += 1
        ents = canon_to_entities.get(p["canonical"], set())
        if not ents:
            rp_missing += 1
            if p["replace"]:
                leaks += 1              # nothing to redact means nothing IS redacted
            else:
                rp_correct += 1         # gold says keep; an absent entity keeps it
            continue
        pred_replace = not any(
            by_id[eid].attributes.get("replace", True) is False for eid in ents)
        if pred_replace == p["replace"]:
            rp_correct += 1
        elif p["replace"] and not pred_replace:
            leaks += 1
        else:
            over_red += 1
    R["replace"] = {"total": rp_total, "correct": rp_correct, "leaks": leaks,
                    "over_redactions": over_red, "missing": rp_missing,
                    "accuracy": _acc(rp_correct, rp_total),
                    **_balance(gold["people"], "replace")}

    # ---------- DATES ----------
    # Read `resolved_value` off the pipeline's own date entities, so the score
    # reflects whatever the second line settled on: the rule's parse, a checked LLM
    # fill of a rule miss, or nothing (rejected / both layers blind).
    #
    # A gold row the pipeline built NO entity for used to `continue`, dropping it
    # from the numerator AND the denominator -- an unresolvable date simply left the
    # measurement. It is a miss: count it, and report `no_entity` so the two kinds
    # of failure (built it and got it wrong / never built it) stay separable.
    date_pool = ents_of("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR", "DATE_OF_BIRTH")
    d_total = d_pass = d_noent = 0
    d_fail, d_by_action = [], Counter()
    for d in gold.get("dates", []):
        d_total += 1
        ent = entity_for(d["text"], date_pool, d.get("context"))
        if ent is None:
            d_noent += 1
            d_by_action["no-entity"] += 1
            d_fail.append((d["text"], "NO ENTITY", d.get("resolved") or "FLAG",
                           "no-entity"))
            continue
        got = ent.attributes.get("resolved_value")
        act = (ledger.get(ent.entity_id, {}).get("resolved_value"))
        d_by_action[act.action if act else "n/a"] += 1
        if d.get("expect_flag"):
            ok = got is None
        else:
            ok = got is not None and abs(
                (dateparser.parse(got).date()
                 - dateparser.parse(d["resolved"]).date()).days) <= d.get("tolerance_days", 0)
        d_pass += 1 if ok else 0
        if not ok:
            d_fail.append((d["text"], got, d.get("resolved") or "FLAG",
                           act.action if act else "n/a"))
    R["dates"] = {"total": d_total, "pass": d_pass, "no_entity": d_noent,
                  "accuracy": _acc(d_pass, d_total), "fail": d_fail,
                  "actions": dict(d_by_action)}

    # ---------- AGES ----------
    # `expect_flag` mirrors the date convention: the row asserts that NOTHING should
    # resolve here, because the span is not an age at all (a measurement the detector
    # typed AGE). Without it the corpus could not carry the negative case, so
    # `checks/ages.not_a_measurement` doing its job was invisible to every number.
    age_pool = ents_of("AGE")
    a_total = a_pass = a_noent = 0
    a_fail, a_by_action = [], Counter()

    def _age_hit(e, row):
        """Does this entity satisfy `row`, within its stated tolerance?

        Used to pick the BEST candidate rather than the first, which is legitimate
        where a gold row describes every occurrence of its surface form. Where the
        occurrences DISAGREE the row carries a `context` instead, and
        `entities_for` has already narrowed the pool to one.
        """
        v = e.attributes.get("value")
        if row.get("expect_flag"):
            return v is None
        return v is not None and abs(v - row["value"]) <= row.get("tolerance", 0)

    for a in gold.get("ages", []):
        a_total += 1
        cands = entities_for(a["text"], age_pool, a.get("context"))
        if not cands:
            a_noent += 1
            a_by_action["no-entity"] += 1
            a_fail.append((a["text"], "NO ENTITY",
                           "FLAG" if a.get("expect_flag") else a["value"], "no-entity"))
            continue
        ent = next((e for e in cands if _age_hit(e, a)), cands[0])
        got = ent.attributes.get("value")
        act = ledger.get(ent.entity_id, {}).get("value")
        a_by_action[act.action if act else "n/a"] += 1
        ok = _age_hit(ent, a)
        a_pass += 1 if ok else 0
        if not ok:
            a_fail.append((a["text"], got,
                           "FLAG" if a.get("expect_flag") else a["value"],
                           act.action if act else "n/a"))
    # ---------- SHIFTABLE / APPROXIMATE / KIND / STATED_WITH ----------
    # The four remaining arbitrated fields with no gold. All four feed the DATE
    # SHIFTER, which is the next stage: `shiftable` says whether a date may move at
    # all, `approximate` says how precisely it may be re-stated, `stated_with` is
    # the age-arithmetic constraint that has to survive the move, and `kind` is
    # what an identifier gets replaced WITH.
    def _row_attr(rows, pool, key, get, match=None, canon=None, default=None):
        """Three-way score of one attribute over text-keyed gold rows.

        Same three outcomes as `_person_attr`, plus `no_entity` -- a row the
        pipeline built nothing for, which is a miss and not an absence.

        `default` is the value an ABSENT attribute means. Some fields here are
        written only in one direction on purpose -- `rules/ages.py` sets
        `approximate=True` for a vague expression and leaves a precise one with no
        key at all -- so reading absence as "missing" would score the pipeline's own
        convention as a failure. Where absence really does mean "nothing decided"
        (`shiftable` is always written), leave `default` as None so a miss is a miss.
        """
        canon = canon or (lambda v: v if v is None else bool(v)
                          if isinstance(v, bool) else str(v).strip().lower())
        out = {"total": 0, "correct": 0, "wrong": 0, "missing": 0,
               "no_entity": 0, "fail": []}
        for row in rows:
            if key not in row:
                continue
            out["total"] += 1
            cands = entities_for(row["text"], pool, row.get("context"))
            if not cands:
                out["no_entity"] += 1
                out["missing"] += 1
                out["fail"].append((row["text"], "NO ENTITY", row[key]))
                continue
            ent = (next((e for e in cands if match(e, row)), cands[0])
                   if match else cands[0])
            raw = get(ent)
            if raw is None:
                raw = default
            want, got = canon(row[key]), canon(raw)
            if got is None and want is not None:
                out["missing"] += 1
                out["fail"].append((row["text"], None, row[key]))
            elif got == want:
                out["correct"] += 1
            else:
                out["wrong"] += 1
                out["fail"].append((row["text"], raw, row[key]))
        out["accuracy"] = _acc(out["correct"], out["total"])
        return out

    R["shiftable"] = _row_attr(gold.get("dates", []), date_pool, "shiftable",
                               lambda e: e.attributes.get("shiftable"))
    R["shiftable"].update(_balance(gold.get("dates", []), "shiftable"))

    # One score over dates AND ages: it is the same question ("is this value
    # rounded?") and the same policy answers it for both.
    #
    # `default=False` is what a CONSUMER sees, and the two ways this field fails
    # look identical from there but are not the same bug:
    #   * a bare year ("1921", "1948") gets NO `approximate` key at all, so its
    #     January-1st placeholder reads as a day-precise date;
    #   * an anchor naming an era ("the Great Recession") gets `approximate=False`
    #     ASSERTED, by `anchor_in_table_is_exact` -- true for Katrina and 9/11, and
    #     wrong for a multi-year recession pinned to one proxy day.
    # The first is an abstention that defaults the wrong way; the second is a
    # confident wrong answer.
    appr_d = _row_attr(gold.get("dates", []), date_pool, "approximate",
                       lambda e: e.attributes.get("approximate"), default=False)
    appr_a = _row_attr(gold.get("ages", []), age_pool, "approximate",
                       lambda e: e.attributes.get("approximate"), match=_age_hit,
                       default=False)
    R["approximate"] = {k: appr_d[k] + appr_a[k]
                        for k in ("total", "correct", "wrong", "missing",
                                  "no_entity", "fail")}
    R["approximate"]["accuracy"] = _acc(R["approximate"]["correct"],
                                        R["approximate"]["total"])
    R["approximate"].update(_balance(gold.get("dates", []) + gold.get("ages", []),
                                     "approximate"))

    # `kind` is handed to the pipeline by the detector, so this is a REGRESSION
    # GUARD on the arbitration: it catches the second line overwriting a type the
    # detector already got right, which is the only way this field can go wrong.
    id_kind_pool = ents_of(*ID_CATS)
    R["kind"] = _row_attr(gold.get("identifiers", []), id_kind_pool, "type",
                          lambda e: e.attributes.get("kind") or e.category)

    # STATED_WITH lives on an edge, not in `attributes`, so it is read from the
    # edge list. Gold names the date THIS transcript ties the age to, or null --
    # and null is the common case, which is the point: the rule links every age to
    # the nearest date within one sentence whether or not the sentence ties them
    # together, so the null rows are where its false positives land. A gold LIST
    # means two expressions name the same moment and either answer is correct.
    sw_by_age = {}
    for ed in edges:
        if ed.relation == Relation.STATED_WITH:
            tgt = by_id.get(ed.target)
            sw_by_age.setdefault(ed.source, set()).update(
                m.text for m in (tgt.mentions if tgt else []))
    sw_total = sw_correct = sw_wrong = sw_missing = 0
    sw_fail = []
    for a in gold.get("ages", []):
        if "stated_with" not in a:
            continue
        sw_total += 1
        cands = entities_for(a["text"], age_pool, a.get("context"))
        ent = next((e for e in cands if _age_hit(e, a)), cands[0] if cands else None)
        got = sorted(sw_by_age.get(ent.entity_id, set())) if ent else []
        want = a["stated_with"]
        want = [] if want is None else ([want] if isinstance(want, str) else want)
        want_l = {w.lower() for w in want}
        if not want:
            if not got:
                sw_correct += 1
            else:
                sw_wrong += 1                       # a pairing gold says is not there
                sw_fail.append((a["text"], "/".join(got), None))
        elif got and {g.lower() for g in got} & want_l:
            sw_correct += 1
        elif not got:
            sw_missing += 1
            sw_fail.append((a["text"], None, "|".join(want)))
        else:
            sw_wrong += 1
            sw_fail.append((a["text"], "/".join(got), "|".join(want)))
    R["stated_with"] = {"total": sw_total, "correct": sw_correct, "wrong": sw_wrong,
                        "missing": sw_missing, "accuracy": _acc(sw_correct, sw_total),
                        "gold_pos": sum(1 for a in gold.get("ages", [])
                                        if a.get("stated_with")),
                        "gold_neg": sum(1 for a in gold.get("ages", [])
                                        if "stated_with" in a and not a["stated_with"]),
                        "fail": sw_fail}

    R["ages"] = {"total": a_total, "pass": a_pass,
                 "accuracy": _acc(a_pass, a_total), "fail": a_fail,
                 "actions": dict(a_by_action)}

    # ---------- LOCATIONS ----------
    # TYPE CORRECTNESS, not type presence. This used to count a place as correctly
    # typed whenever `entity.subtype` was merely TRUTHY, and report the result as
    # "location typing accuracy" -- so `Washington` typed STATE (the transcript
    # means D.C.), `Mingo County` typed CITY and `Guanajuato` typed CITY all scored
    # as right answers, and the headline read 100% while a fifth of the types were
    # wrong. Both sides now fold through `loc_buckets._canon_type`, which is the
    # harness's own vocabulary rather than the pipeline's.
    #
    # `typed` is kept as a separate count because the two failures are different:
    # NO type means the second line abstained (and `keep_rests_on_a_verified_type`
    # will refuse to keep the name, over-redacting it), while a WRONG type is a
    # confident false answer that can force the opposite decision.
    loc_pool = ents_of("LOCATION", "INSTITUTION")
    gaz_gold = [l for l in gold.get("locations", []) if l.get("in_gazetteer")]
    off_gaz = [l for l in gold.get("locations", []) if not l.get("in_gazetteer")]

    def _typing(rows):
        """(typed, correct, wrong, failures) over one group of gold location rows."""
        typed = correct = wrong = 0
        fails = []
        for l in rows:
            ent = entity_for(l["text"], loc_pool)
            got = _canon_type(getattr(ent, "subtype", None) if ent else None)
            if got is None:
                fails.append((l["text"], None, l["type"]))
                continue
            typed += 1
            if got == _canon_type(l["type"]):
                correct += 1
            else:
                wrong += 1
                fails.append((l["text"], ent.subtype, l["type"]))
        return typed, correct, wrong, fails

    typed, typed_ok, typed_bad, typed_fail = _typing(gaz_gold)
    typed_off, off_ok, off_bad, off_fail = _typing(off_gaz)

    # LOCATION_PARENT. `resolve_all` arbitrates this field and `parent_resolves`
    # promotes it into a LOCATED_IN edge, but the harness only ever printed the edge
    # COUNT -- so a transcript that produced zero hierarchy (interview_002,
    # rules-only) looked the same as one that produced the right hierarchy. Gold now
    # names the containing place for every location whose parent this transcript
    # also mentions, and the edge is checked end to end.
    #
    # SCORED AGAINST THE ANCESTOR CHAIN, not the immediate parent alone. "Matewan is
    # in West Virginia" is TRUE, just coarser than "Matewan is in Mingo County", and
    # grading it as a miss AND a false positive would be the harness manufacturing
    # two errors out of one correct-but-less-specific answer. So an edge to any true
    # ancestor counts, `exact` records how many hit the immediate parent, and only an
    # edge to a place that is NOT an ancestor is a false positive.
    #
    # An edge whose source has no gold parent at all is UNSCORED rather than wrong:
    # gold declined to assert a container there (Tug Fork straddles two states), and
    # silence in gold is not evidence against the pipeline.
    n_loc_edges = sum(1 for ed in edges if ed.relation == Relation.LOCATED_IN)
    n_loc_edges_llm = sum(1 for ed in edges if ed.relation == Relation.LOCATED_IN
                          and "llm" in str(ed.evidence))
    loc_eid = {}                        # gold location text -> entity_id
    for l in gold.get("locations", []):
        ent = entity_for(l["text"], loc_pool)
        if ent is not None:
            loc_eid[l["text"]] = ent.entity_id
    eid2text = {v: k for k, v in loc_eid.items()}
    gold_parent_of = {l["text"]: l.get("parent") for l in gold.get("locations", [])}

    def _ancestors(name):
        """Every containing place gold puts above `name`, nearest first."""
        out, seen, cur = [], {name}, gold_parent_of.get(name)
        while cur and cur not in seen:
            seen.add(cur)
            out.append(cur)
            cur = gold_parent_of.get(cur)
        return out

    pred_par = [(ed.source, ed.target) for ed in edges
                if ed.relation == Relation.LOCATED_IN]
    par_scorable = par_exact = par_hit = 0
    par_miss, par_fp, par_coarse, par_unscored = [], [], [], []
    for l in gold.get("locations", []):
        if not l.get("parent") or l["text"] not in loc_eid:
            continue
        chain = [a for a in _ancestors(l["text"]) if a in loc_eid]
        if not chain:                   # no ancestor has a node to point at
            continue
        par_scorable += 1
        src = loc_eid[l["text"]]
        got = {eid2text.get(t) for s, t in pred_par if s == src}
        if l["parent"] in got:
            par_exact += 1
            par_hit += 1
        elif got & set(chain):
            par_hit += 1
            par_coarse.append(f"{l['text']}->{sorted(got & set(chain))[0]} "
                              f"(want {l['parent']})")
        else:
            par_miss.append(f"{l['text']}->{l['parent']}")
    for s, t in pred_par:
        sname, tname = eid2text.get(s), eid2text.get(t)
        if sname is None or not gold_parent_of.get(sname):
            par_unscored.append(f"{sname or s}->{tname or t}")
        elif tname not in _ancestors(sname):
            par_fp.append(f"{sname}->{tname or t}")
    R["loc_parent"] = {
        "gold": sum(1 for l in gold.get("locations", []) if l.get("parent")),
        "scorable": par_scorable, "pred": len(pred_par),
        "scored_pred": len(pred_par) - len(par_unscored),
        "tp": par_hit, "exact": par_exact, "fp": len(par_fp),
        "unscored": len(par_unscored),
        "precision": _acc(par_hit, len(pred_par) - len(par_unscored)),
        "recall": _acc(par_hit, par_scorable),
        "misses": sorted(par_miss), "false_pos": sorted(par_fp),
        "coarse": sorted(par_coarse), "unscored_edges": sorted(par_unscored),
    }

    # PLACE REDACTION. Nothing scored this, and it is the highest-risk category for
    # re-identifying a rural interviewee -- "Red Jacket" plus an age plus "miner"
    # identifies one household. The headline "privacy leaks" number counted PEOPLE
    # only, so a kept village was invisible to every metric in this harness.
    lr_total = lr_correct = lr_leaks = lr_over = lr_noent = 0
    for l in gold.get("locations", []):
        if "replace" not in l:
            continue
        lr_total += 1
        ent = entity_for(l["text"], loc_pool)
        if ent is None:
            # no entity -> nothing downstream redacts the span, so a gold `replace`
            # is a leak. Counted rather than skipped: dropping the row hid the
            # failure from the numerator and the denominator both.
            lr_noent += 1
            if l["replace"]:
                lr_leaks += 1
            else:
                lr_correct += 1
            continue
        # absent `replace` means nothing decided -> a consumer keying off it keeps
        # the name, so score it as a keep
        got = ent.attributes.get("replace", False) is True
        if got == l["replace"]:
            lr_correct += 1
        elif l["replace"] and not got:
            lr_leaks += 1
        else:
            lr_over += 1

    R["locations"] = {"in_gaz": len(gaz_gold), "typed": typed,
                      "type_ok": typed_ok, "type_wrong": typed_bad,
                      "accuracy": _acc(typed_ok, len(gaz_gold)),
                      "off_gaz": len(off_gaz), "typed_off_gaz": typed_off,
                      "off_type_ok": off_ok, "off_type_wrong": off_bad,
                      "off_accuracy": _acc(off_ok, len(off_gaz)),
                      "type_fail": typed_fail + off_fail,
                      "edges": n_loc_edges, "edges_llm": n_loc_edges_llm,
                      "rep_total": lr_total, "rep_correct": lr_correct,
                      "rep_leaks": lr_leaks, "rep_over": lr_over,
                      "rep_missing": lr_noent,
                      "rep_accuracy": _acc(lr_correct, lr_total),
                      **_balance(gold.get("locations", []), "replace")}

    # ---------- REDACTION OF DATES AND AGES ----------
    # `replace_date` / `replace_age` decide whether a temporal span's surface text
    # survives. Nothing scored them because until recently nothing DECIDED them --
    # AGE was the one category in the graph with no redaction directive at all, so a
    # consumer keying off `replace` printed the speaker's ages verbatim.
    #
    # Scored with the same two-direction split as people and places, because the
    # errors are not interchangeable: a LEAK (should replace, kept) is unrecoverable,
    # an over-redaction costs narrative colour.
    def _redaction(gold_rows, pool, match=None):
        """Score `replace` on a group of gold rows, splitting errors by direction.

        Shared by the date and age redaction scores. Returns counts of correct
        decisions plus LEAKS (gold says replace, pipeline kept it -- unrecoverable)
        and over-redactions (gold says keep, pipeline replaced it -- costs only
        colour).

        `match` is an optional predicate for picking WHICH candidate entity a gold
        row is about, needed because gold is text-keyed while AGE entities are
        per-mention; without it the first candidate is graded. A row carrying a
        `context` has already been narrowed to one candidate before `match` runs.

        A row with no matching entity is a LEAK when gold says replace: nothing
        downstream redacts a span no entity covers. It used to be skipped, which
        removed it from both halves of the ratio.
        """
        total = correct = leaks = over = noent = 0
        fails = []
        for row in gold_rows:
            if "replace" not in row:
                continue
            total += 1
            cands = entities_for(row["text"], pool, row.get("context"))
            if not cands:
                noent += 1
                if row["replace"]:
                    leaks += 1
                    fails.append((row["text"], "NO ENTITY", "replace"))
                else:
                    correct += 1
                continue
            # gold is text-keyed and AGE entities are per-mention, so grade the span
            # the gold row is actually about (`match` picks it) rather than whichever
            # one happens to come first.
            ent = (next((e for e in cands if match(e, row)), cands[0])
                   if match else cands[0])
            got = ent.attributes.get("replace", False) is True
            if got == row["replace"]:
                correct += 1
            elif row["replace"] and not got:
                leaks += 1
                fails.append((row["text"], "KEPT", "replace"))
            else:
                over += 1
                fails.append((row["text"], "replaced", "keep"))
        return {"total": total, "correct": correct, "leaks": leaks, "over": over,
                "missing": noent, "accuracy": _acc(correct, total), "fail": fails,
                **_balance(gold_rows, "replace")}

    R["date_redaction"] = _redaction(gold.get("dates", []), date_pool)
    R["age_redaction"] = _redaction(gold.get("ages", []), age_pool,
                                    match=_age_hit)

    # ---------- IDENTIFYING OCCUPATIONS ----------
    # The field that measured nothing before it had a rule layer and checkers: the
    # model called seven of nine occupations "identifying", which is a signal that
    # fires on everything.
    #
    # READ THE CLASS BALANCE, NOT THE PERCENTAGE. Every occupation in all three
    # transcripts is a common job, so gold is all-False -- `gold_pos` is 0 and a
    # constant "False" scores 100%. That makes this a REGRESSION GUARD (the rule
    # table plus `identifying_not_a_common_occupation` still hold the line) and not
    # an accuracy. The corpus contains no rare occupation to supply the other class,
    # so the honest fix is to say so in the output rather than to print a number
    # that cannot fall; `report._fmt_bal` marks it.
    occ_pool = ents_of("OCCUPATION")
    id_total = id_correct = id_over = id_noent = 0
    id_fail = []
    for x in gold.get("identifiers", []):
        if "identifying" not in x:
            continue
        id_total += 1
        ent = entity_for(x["text"], occ_pool)
        if ent is None:
            id_noent += 1
            id_over += 1
            id_fail.append((x["text"], "NO ENTITY", x["identifying"]))
            continue
        got = ent.attributes.get("identifying") is True
        if got == x["identifying"]:
            id_correct += 1
        else:
            id_over += 1
            id_fail.append((x["text"], got, x["identifying"]))
    R["identifying"] = {"total": id_total, "correct": id_correct,
                        "wrong": id_over, "missing": id_noent,
                        "accuracy": _acc(id_correct, id_total),
                        "fail": id_fail,
                        **_balance(gold.get("identifiers", []), "identifying")}

    # ---------- OWNERSHIP ----------
    # THE field interviewee-only surrogate generation runs on: which spans belong to
    # the subject. It was resolved, blocked on, and never scored.
    #
    # Counted three ways, because the two error directions are not
    # interchangeable. A WRONG owner builds the subject's surrogate identity out of
    # somebody else's data (or hands their data to a third party); an UNRESOLVED owner
    # is recoverable -- it blocks and a human decides. So `wrong` is the number to
    # drive to zero, and `missing` is the review burden.
    #
    # A row with no matching entity counts as UNRESOLVED, not as nothing: no entity
    # means no owner reaches the artifact, which is the same review burden as an
    # owner the pipeline declined to decide.
    own_pool = ents_of(*(ID_CATS + ("AGE", "DATE_OF_BIRTH")))
    o_total = o_correct = o_wrong = o_missing = o_noent = 0
    o_fail = []
    gold_owners = [(x["text"], x["owner"], x.get("context"))
                   for x in gold.get("identifiers", []) if x.get("owner")]
    gold_owners += [(a["text"], a["owner"], a.get("context"))
                    for a in gold.get("ages", []) if a.get("owner")]
    gold_owners += [(d["text"], d["owner"], d.get("context"))
                    for d in gold.get("dates", []) if d.get("owner")]
    for text_value, want, ctx in gold_owners:
        o_total += 1
        cands = entities_for(text_value, own_pool, ctx)
        if not cands:
            o_noent += 1
            o_missing += 1
            o_fail.append((text_value, "NO ENTITY", want))
            continue
        got_set = {e.attributes.get("owner") for e in cands}
        got_set.discard(None)
        if not got_set:
            o_missing += 1
            o_fail.append((text_value, None, want))
        elif got_set == {want}:
            o_correct += 1
        else:
            o_wrong += 1
            o_fail.append((text_value, "/".join(sorted(got_set)), want))
    R["owner"] = {"total": o_total, "correct": o_correct, "wrong": o_wrong,
                  "missing": o_missing, "no_entity": o_noent,
                  "accuracy": _acc(o_correct, o_total), "fail": o_fail}

    # ---------- THE INTERVIEWEE ----------
    # The subject's OWN attributes, which no metric touched: `gender recall` iterates
    # `gold["people"]`, i.e. third parties only. This is the block that tells you
    # whether the next stage can mint a surrogate for the person you are actually
    # de-identifying.
    iv_gold = gold.get("interviewee") or {}
    iv_res = {}
    if iv_gold:
        ivg = interviewee.attributes.get("gender")
        want_g = iv_gold.get("gender")
        iv_res["gender"] = ("correct" if ivg and ivg == want_g else
                            "missing" if not ivg else "wrong")
        ive = interviewee.attributes.get("ethnicity")
        want_e = iv_gold.get("ethnicity")
        iv_res["ethnicity"] = ("n/a" if not want_e else
                               "correct" if ive and str(ive).lower() == want_e else
                               "missing" if not ive else "wrong")
        # identity: gold `null` means the transcript never names the speaker, so
        # abstaining is the CORRECT answer and naming anyone is a hard error
        got_id = interviewee.attributes.get("identity_entity_id")
        want_id = iv_gold.get("identity")
        if want_id is None:
            iv_res["identity"] = "correct" if not got_id else "wrong"
        else:
            got_forms = {f.lower() for f in interviewee.sorted_mentions}
            iv_res["identity"] = ("correct" if want_id.lower() in got_forms else
                                  "missing" if not got_id else "wrong")
        # The speaker's own NAME PARTS -- the two fields a surrogate identity for
        # the person being de-identified is actually built from. `gold` is null/null
        # where the transcript never names the speaker, so abstaining is correct
        # there and inventing a name is a hard error.
        for part in ("given_name", "surname"):
            if part not in iv_gold:
                continue
            want_p = iv_gold[part]
            got_p = interviewee.attributes.get(part)
            want_c = want_p.lower() if want_p else None
            got_c = str(got_p).strip().lower() if got_p and str(got_p).strip() else None
            iv_res[part] = ("correct" if got_c == want_c else
                            "missing" if got_c is None else "wrong")
        want_dob = iv_gold.get("dob")
        got_dob = None
        for e in ents_of("DATE_OF_BIRTH"):
            if e.attributes.get("owner") == "interviewee":
                got_dob = e.attributes.get("resolved_value")
                break
        iv_res["dob"] = ("n/a" if not want_dob else
                         "correct" if got_dob == want_dob else
                         "missing" if not got_dob else "wrong")
        iv_res["values"] = {"gender": ivg, "ethnicity": ive, "dob": got_dob,
                            "identity": got_id,
                            "given_name": interviewee.attributes.get("given_name"),
                            "surname": interviewee.attributes.get("surname")}
    R["interviewee"] = iv_res

    # ---------- SECOND LINE ----------
    # Reported so the arbitration is visible and a regression in it is obvious. The
    # decisions themselves are now scored above: `owner` against gold `owner` fields,
    # the subject's own attributes against the gold `interviewee` block, and place
    # redaction against gold `replace`.
    # `reject` splits into two very different outcomes: the LLM proposed something
    # and a deterministic checker REFUTED it, or neither layer produced a value at
    # all. Collapsing them hides which half of the pipeline is missing.
    def _label(res):
        """The action label for the summary, splitting `reject` in two.

        A rejection has two very different causes: the LLM proposed something and a
        deterministic checker REFUTED it, or neither layer produced a value at all
        ("blind"). Collapsing them hides which half of the pipeline is missing --
        refutations mean the checkers are working, blind rows mean nothing tried.
        """
        if res.action != "reject":
            return res.action
        return "refuted" if res.checks_failed else "blind"

    actions, by_field = Counter(), {}
    for eid, fields in ledger.items():
        if eid == "_edges":
            continue
        for fname, res in fields.items():
            # relations and same-person claims are stored per partner
            # (`relation:<other_id>`, `same_person:<other_id>`); collapse them to one
            # row each so the summary stays readable
            key = fname.split(":", 1)[0] if ":" in fname else fname
            actions[_label(res)] += 1
            by_field.setdefault(key, Counter())[_label(res)] += 1
    iv_g = ledger.get(interviewee.entity_id, {}).get("interviewee_gender")
    iv_id = ledger.get(interviewee.entity_id, {}).get("interviewee_identity")
    R["second_line"] = {
        "llm_ran": info.get("llm_ran", False),
        "llm_model": info.get("llm_model"),
        "actions": dict(actions),
        "by_field": {k: dict(v) for k, v in by_field.items()},
        "blocking": info.get("blocking", []),
        "interviewee_gender": interviewee.attributes.get("gender"),
        "interviewee_gender_action": iv_g.action if iv_g else "n/a",
        "interviewee_gender_reason": (iv_g.reason if iv_g else ""),
        "interviewee_name": " / ".join(interviewee.sorted_mentions) or None,
        "interviewee_identity_action": iv_id.action if iv_id else "n/a",
        "interviewee_identity_reason": (iv_id.reason if iv_id else ""),
        "interviewee_identity_checks": list(iv_id.checks_passed) if iv_id else [],
    }
    return R
