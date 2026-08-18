"""
Evaluation for the knowledge-graph stage. It simulates a perfect detector from the
gold annotations and runs those spans through `graph.pipeline.run_pipeline` -- THE
SAME entry point the demos and the report use.

That matters: this harness used to re-assemble the stages by hand, calling
merge/coref/kinship/attributes directly and resolving dates and ages into throwaway
Entity objects. It therefore never exercised `openworld_propose`,
`identifier_judge_pass`, `infer_interviewee_gender`, `build_identifier_entities`,
`_link_interviewee_pii`, `age_date_constraints`, or -- once the layers were unified
-- `graph.second_line.resolve_all`. Every number it printed for dates, ages,
gender, replace and locations described a pipeline that was never shipped. Now the
numbers come from the real one, so a value the second line CONFIRMED, FILLED after
its checks, or REJECTED is scored exactly as a consumer would see it.

Relations are still broken down by provenance (rule vs llm) so the LLM path's
recall gain and precision cost are both visible.

Set KG_USE_LLM=1 (with Ollama up) to score rules+LLM; leave it unset for the
rules-only baseline. EVAL_NO_COREF=1 skips the coref stage.
"""

from __future__ import annotations
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

# quiet fastcoref / transformers / datasets chatter before importing them
for _n in ("fastcoref", "transformers", "datasets", "urllib3"):
    logging.getLogger(_n).setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from dateutil import parser as dateparser
from graph.loader import resolve_overlaps, make_mentions
from graph.pipeline import run_pipeline
from graph.models import Relation
from graph.identifiers import ID_CATS

RUN_COREF = os.environ.get("EVAL_NO_COREF") != "1"

# opt-in local LLM adjudicator (KG_USE_LLM=1); no-ops if Ollama isn't running
_LLM = None
if os.environ.get("KG_USE_LLM") == "1":
    from llm_layer import default_client
    _LLM = default_client()

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "tests"                       # transcripts + gold live here
DATA_GAZ = REPO / "data" / "gazetteer.csv"

# kin-word synonyms -> canonical family term, for scoring relation "detail"
_KIN_CANON = {}
for _base, _variants in {
    "mother": "mother mom mommy mum mummy mama mamma momma ma",
    "father": "father dad daddy papa poppa pop pops pa",
    # `mamaw` and friends: absent here, the pipeline's correct "mamaw" edge for
    # interview_002's grandmother scored as a WRONG DETAIL against gold's
    # "grandmother" -- the same table gap that kept `graph/kinship.py` from
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
    d = (detail or "").strip().lower()
    return _KIN_CANON.get(d, d)


def _find_spans(text: str, surface: str):
    """All occurrences of `surface` at letter boundaries."""
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(surface) + r"(?![A-Za-z])")
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def _build_detections(text: str, gold: dict):
    """Simulated perfect detector: gold surface forms -> detection dicts."""
    dets = []

    def add(surface, etype):
        for s, e in _find_spans(text, surface):
            dets.append({"text": surface, "start": s, "end": e,
                         "entity_type": etype, "score": 1.0})

    for p in gold["people"]:
        for form in p["forms"]:
            add(form, "PERSON")
    # The speaker's own name, when the transcript states it -- see the same note in
    # `demo_utils.detections_from_gold`. Kept out of `people` so clustering and
    # relation scoring still describe THIRD PARTIES only.
    for form in (gold.get("interviewee") or {}).get("forms", []):
        add(form, "PERSON")
    for loc in gold.get("locations", []):
        add(loc["text"], "LOCATION")
    for d in gold.get("dates", []):
        add(d["text"], d["category"])
    for a in gold.get("ages", []):
        add(a["text"], "AGE")
    # Direct identifiers were previously omitted, so the harness never built any
    # identifier entities -- which meant no ownership resolution and no identifier
    # rows in the second-line ledger.
    for x in gold.get("identifiers", []):
        add(x["text"], x["type"])
    return dets


def _acc(num, den):
    return (num / den) if den else None


def _fmt(x):
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def _relation_report(gold_rel: dict, pred_rel: dict, pred_src: dict) -> dict:
    """Score predicted relations against gold and break the result down by which
    layer produced each edge. `pred_src` maps a (source, target) pair to 'rule' or
    'llm'. `llm_gain` is the set of gold relations the LLM found that the rules
    missed -- the recall the LLM path actually buys."""
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
        return [e for e in entities if e.category in categories]

    def entities_for(text_value: str, pool):
        """EVERY pipeline entity whose mentions include this gold surface form.

        Gold is keyed by surface TEXT with no offsets, so one gold row can legitimately
        correspond to several entities: AGE entities are one-per-mention (see
        `pipeline._simple_entities`), so "twelve" in "the water came up twelve feet"
        and "twelve" in "my daughter Trang was maybe twelve" are two entities. Picking
        the first and scoring only that graded whichever span happened to come first in
        the transcript -- and marked the age WRONG when the leading span was the
        measurement `checks/ages.not_a_measurement` had correctly refused.
        """
        t = text_value.lower()
        return [e for e in pool if any(m.text.lower() == t for m in e.mentions)]

    def entity_for(text_value: str, pool):
        """The first such entity, for the scorers that need exactly one."""
        found = entities_for(text_value, pool)
        return found[0] if found else None

    # ---- surface form -> gold canonical ----
    form2canon = {}
    for p in gold["people"]:
        for form in p["forms"]:
            form2canon[form.lower()] = p["canonical"]

    def canon_of_entity(e):
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

    # ---------- REPLACE / SAFETY ----------
    rp_total = rp_correct = leaks = over_red = 0
    for p in gold["people"]:
        rp_total += 1
        ents = canon_to_entities.get(p["canonical"], set())
        pred_replace = True
        if ents:
            pred_replace = not any(
                by_id[eid].attributes.get("replace", True) is False for eid in ents)
        if pred_replace == p["replace"]:
            rp_correct += 1
        elif p["replace"] and not pred_replace:
            leaks += 1
        else:
            over_red += 1
    R["replace"] = {"total": rp_total, "correct": rp_correct, "leaks": leaks,
                    "over_redactions": over_red, "accuracy": _acc(rp_correct, rp_total)}

    # ---------- DATES ----------
    # Read `resolved_value` off the pipeline's own date entities, so the score
    # reflects whatever the second line settled on: the rule's parse, a checked LLM
    # fill of a rule miss, or nothing (rejected / both layers blind).
    date_pool = ents_of("DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR", "DATE_OF_BIRTH")
    d_total = d_pass = 0
    d_fail, d_by_action = [], Counter()
    for d in gold.get("dates", []):
        ent = entity_for(d["text"], date_pool)
        if ent is None:
            continue
        got = ent.attributes.get("resolved_value")
        act = (ledger.get(ent.entity_id, {}).get("resolved_value"))
        d_total += 1
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
    R["dates"] = {"total": d_total, "pass": d_pass,
                  "accuracy": _acc(d_pass, d_total), "fail": d_fail,
                  "actions": dict(d_by_action)}

    # ---------- AGES ----------
    age_pool = ents_of("AGE")
    a_total = a_pass = 0
    a_fail, a_by_action = [], Counter()
    for a in gold.get("ages", []):
        cands = entities_for(a["text"], age_pool)
        if not cands:
            continue
        a_total += 1

        # gold is text-keyed and AGE entities are per-mention, so score the BEST
        # matching entity: the gold row asserts "this expression means N years", and
        # the pipeline satisfies it if any span of that expression resolved to N.
        def _val(e):
            return e.attributes.get("value")

        def _hit(e):
            v = _val(e)
            return v is not None and abs(v - a["value"]) <= a.get("tolerance", 0)

        ent = next((e for e in cands if _hit(e)), cands[0])
        got = _val(ent)
        act = ledger.get(ent.entity_id, {}).get("value")
        a_by_action[act.action if act else "n/a"] += 1
        ok = _hit(ent)
        a_pass += 1 if ok else 0
        if not ok:
            a_fail.append((a["text"], got, a["value"], act.action if act else "n/a"))
    R["ages"] = {"total": a_total, "pass": a_pass,
                 "accuracy": _acc(a_pass, a_total), "fail": a_fail,
                 "actions": dict(a_by_action)}

    # ---------- LOCATIONS ----------
    # Gazetteer typing as before, plus the places the gazetteer does NOT know: those
    # now get a type only if the LLM proposed one AND it cleared the enum/parent
    # checks, so the second column is the second line's actual contribution.
    loc_pool = ents_of("LOCATION", "INSTITUTION")
    gaz_gold = [l for l in gold.get("locations", []) if l.get("in_gazetteer")]
    off_gaz = [l for l in gold.get("locations", []) if not l.get("in_gazetteer")]
    typed = sum(1 for l in gaz_gold
                if (entity_for(l["text"], loc_pool) or _NoEnt).subtype)
    typed_off = sum(1 for l in off_gaz
                    if (entity_for(l["text"], loc_pool) or _NoEnt).subtype)
    n_loc_edges = sum(1 for ed in edges if ed.relation == Relation.LOCATED_IN)
    n_loc_edges_llm = sum(1 for ed in edges if ed.relation == Relation.LOCATED_IN
                          and "llm" in str(ed.evidence))
    # PLACE REDACTION. Nothing scored this, and it is the highest-risk category for
    # re-identifying a rural interviewee -- "Red Jacket" plus an age plus "miner"
    # identifies one household. The headline "privacy leaks" number counted PEOPLE
    # only, so a kept village was invisible to every metric in this harness.
    lr_total = lr_correct = lr_leaks = lr_over = 0
    for l in gold.get("locations", []):
        if "replace" not in l:
            continue
        ent = entity_for(l["text"], loc_pool)
        if ent is None:
            continue
        lr_total += 1
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
                      "accuracy": _acc(typed, len(gaz_gold)),
                      "off_gaz": len(off_gaz), "typed_off_gaz": typed_off,
                      "edges": n_loc_edges, "edges_llm": n_loc_edges_llm,
                      "rep_total": lr_total, "rep_correct": lr_correct,
                      "rep_leaks": lr_leaks, "rep_over": lr_over,
                      "rep_accuracy": _acc(lr_correct, lr_total)}

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
        total = correct = leaks = over = 0
        fails = []
        for row in gold_rows:
            if "replace" not in row:
                continue
            cands = entities_for(row["text"], pool)
            if not cands:
                continue
            total += 1
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
                "accuracy": _acc(correct, total), "fail": fails}

    R["date_redaction"] = _redaction(gold.get("dates", []), date_pool)
    R["age_redaction"] = _redaction(
        gold.get("ages", []), age_pool,
        match=lambda e, row: e.attributes.get("value") is not None
                             and abs(e.attributes["value"] - row["value"])
                                 <= row.get("tolerance", 0))

    # ---------- IDENTIFYING OCCUPATIONS ----------
    # The field that measured nothing before it had a rule layer and checkers: the
    # model called seven of nine occupations "identifying", which is a signal that
    # fires on everything. Every occupation in both transcripts is a common job, so
    # gold is all-False and this row measures whether the checkers hold that line.
    occ_pool = ents_of("OCCUPATION")
    id_total = id_correct = id_over = 0
    id_fail = []
    for x in gold.get("identifiers", []):
        if "identifying" not in x:
            continue
        ent = entity_for(x["text"], occ_pool)
        if ent is None:
            continue
        id_total += 1
        got = ent.attributes.get("identifying") is True
        if got == x["identifying"]:
            id_correct += 1
        else:
            id_over += 1
            id_fail.append((x["text"], got, x["identifying"]))
    R["identifying"] = {"total": id_total, "correct": id_correct,
                        "wrong": id_over, "accuracy": _acc(id_correct, id_total),
                        "fail": id_fail}

    # ---------- OWNERSHIP ----------
    # THE field interviewee-only surrogate generation runs on: which spans belong to
    # the subject. It was resolved, blocked on, and never scored.
    #
    # Counted three ways, because the two error directions are not
    # interchangeable. A WRONG owner builds the subject's surrogate identity out of
    # somebody else's data (or hands their data to a third party); an UNRESOLVED owner
    # is recoverable -- it blocks and a human decides. So `wrong` is the number to
    # drive to zero, and `missing` is the review burden.
    own_pool = ents_of(*(ID_CATS + ("AGE", "DATE_OF_BIRTH")))
    o_total = o_correct = o_wrong = o_missing = 0
    o_fail = []
    gold_owners = [(x["text"], x["owner"]) for x in gold.get("identifiers", [])
                   if x.get("owner")]
    gold_owners += [(a["text"], a["owner"]) for a in gold.get("ages", [])
                    if a.get("owner")]
    gold_owners += [(d["text"], d["owner"]) for d in gold.get("dates", [])
                    if d.get("owner")]
    for text_value, want in gold_owners:
        cands = entities_for(text_value, own_pool)
        if not cands:
            continue
        o_total += 1
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
                  "missing": o_missing, "accuracy": _acc(o_correct, o_total),
                  "fail": o_fail}

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
                            "identity": got_id}
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


class _NoEnt:
    """Stand-in so a missing entity reads as 'untyped' instead of raising."""
    subtype = None


def _print_one(R):
    c, r, g, rp, d, a, l = (R["cluster"], R["rel"], R["gender"], R["replace"],
                            R["dates"], R["ages"], R["locations"])
    print(f"\n### {R['tid']}")
    print(f"  clustering : recall {_fmt(c['recall'])}  "
          f"(exact {c['exact']}/{c['gold']}, over-merges {c['over_merges']}, "
          f"splits {c['splits']}, coref-merges {c['coref_merges']})")
    print(f"  relations  : P {_fmt(r['precision'])}  R {_fmt(r['recall'])}  "
          f"detail {_fmt(r['detail_acc'])}  (tp {r['tp']}/{r['gold']}, pred {r['pred']})")
    if r["misses"]:
        print(f"      misses    : {r['misses']}")
    if r["false_pos"]:
        print(f"      false pos : {r['false_pos']}")
    if r["bad_detail"]:
        print(f"      bad detail: {r['bad_detail']}")
    if r["llm_pred"]:
        print(f"      llm layer : {r['llm_pred']} edges -> +{r['llm_tp']} correct / "
              f"{r['llm_fp']} false-pos  (recall gain: {r['llm_gain'] or 'none'})")
        if r["llm_false_pos"]:
            print(f"      llm f-pos : {r['llm_false_pos']}")
    print(f"  gender     : recall {_fmt(g['recall'])}  "
          f"(correct {g['correct']}/{g['total']}, wrong {g['wrong']}, missing {g['missing']})")
    print(f"  replace    : acc {_fmt(rp['accuracy'])}  "
          f"(LEAKS {rp['leaks']}, over-redactions {rp['over_redactions']})")
    print(f"  dates      : acc {_fmt(d['accuracy'])}  ({d['pass']}/{d['total']})"
          f"  actions {d['actions']}")
    if d["fail"]:
        for txt, got, want, act in d["fail"]:
            print(f"      fail      : {txt!r} -> {got} (want {want}) [{act}]")
    print(f"  ages       : acc {_fmt(a['accuracy'])}  ({a['pass']}/{a['total']})"
          f"  actions {a['actions']}")
    if a["fail"]:
        for txt, got, want, act in a["fail"]:
            print(f"      fail      : {txt!r} -> {got} (want {want}) [{act}]")
    print(f"  locations  : typed {_fmt(l['accuracy'])}  ({l['typed']}/{l['in_gaz']} in gazetteer)"
          f"  off-gazetteer typed {l['typed_off_gaz']}/{l['off_gaz']}"
          f"  LOCATED_IN {l['edges']} ({l['edges_llm']} llm-verified)")
    if l["rep_total"]:
        print(f"  place redac: acc {_fmt(l['rep_accuracy'])}  "
              f"(LEAKS {l['rep_leaks']}, over-redactions {l['rep_over']})")
    for label, key in (("date redac ", "date_redaction"), ("age redac  ", "age_redaction")):
        b = R[key]
        if not b["total"]:
            continue
        print(f"  {label}: acc {_fmt(b['accuracy'])}  ({b['correct']}/{b['total']}, "
              f"LEAKS {b['leaks']}, over-redactions {b['over']})")
        for txt, got, want in b["fail"]:
            print(f"      fail      : {txt!r} -> {got} (want {want})")
    idg = R["identifying"]
    if idg["total"]:
        print(f"  identifying: acc {_fmt(idg['accuracy'])}  "
              f"({idg['correct']}/{idg['total']} occupations, wrong {idg['wrong']})")
        for txt, got, want in idg["fail"]:
            print(f"      fail      : {txt!r} -> {got} (want {want})")
    o = R["owner"]
    if o["total"]:
        print(f"  ownership  : acc {_fmt(o['accuracy'])}  "
              f"(correct {o['correct']}/{o['total']}, WRONG {o['wrong']}, "
              f"unresolved {o['missing']})")
        for txt, got, want in o["fail"]:
            print(f"      fail      : {txt!r} -> {got} (want {want})")
    iv = R.get("interviewee") or {}
    if iv:
        v = iv["values"]
        print(f"  INTERVIEWEE: gender {iv['gender']} ({v['gender']})  "
              f"ethnicity {iv['ethnicity']} ({v['ethnicity']})  "
              f"identity {iv['identity']}  dob {iv['dob']} ({v['dob']})")

    s = R["second_line"]
    print(f"  2nd line   : llm_ran={s['llm_ran']}"
          + (f" ({s['llm_model']})" if s["llm_model"] else "")
          + f"  {s['actions']}")
    print(f"      interviewee identity: {s['interviewee_name']!r} "
          f"[{s['interviewee_identity_action']}]"
          + (f"  verified by {s['interviewee_identity_checks']}"
             if s["interviewee_identity_checks"] else ""))
    if s["interviewee_identity_reason"]:
        print(f"          reason: {s['interviewee_identity_reason'][:110]}")
    print(f"      interviewee gender: {s['interviewee_gender']!r} "
          f"[{s['interviewee_gender_action']}]"
          + (f"  -> {iv['gender']}" if iv.get("gender") else ""))
    if s["interviewee_gender_reason"] and s["interviewee_gender_action"] != "confirm":
        print(f"          reason: {s['interviewee_gender_reason'][:100]}")
    if s["blocking"]:
        print(f"      BLOCKING ({len(s['blocking'])}):")
        for eid, fname, why in s["blocking"]:
            print(f"          {eid} . {fname}: {why[:90]}")


def _accumulate(agg, R):
    for k in ("cluster", "rel", "gender", "replace", "dates", "ages", "locations",
              "owner", "date_redaction", "age_redaction", "identifying"):
        bucket = agg.setdefault(k, {})
        for f, v in R[k].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                bucket[f] = bucket.get(f, 0) + v
    # the interviewee block is one verdict per field per transcript, so it counts
    # outcomes rather than summing numbers
    iv = agg.setdefault("interviewee", {})
    for f, verdict in (R.get("interviewee") or {}).items():
        if f == "values":
            continue
        iv.setdefault(f, Counter())[verdict] += 1
    sl = agg.setdefault("second_line", {"actions": Counter(), "by_field": {},
                                        "blocking": 0, "llm_ran": False})
    s = R["second_line"]
    sl["llm_ran"] = sl["llm_ran"] or s["llm_ran"]
    sl["llm_model"] = s.get("llm_model") or sl.get("llm_model")
    sl["actions"].update(s["actions"])
    sl["blocking"] += len(s["blocking"])
    for fname, counts in s["by_field"].items():
        sl["by_field"].setdefault(fname, Counter()).update(counts)


def _print_aggregate(agg, n=None):
    c, r, g, rp, d, a, l = (agg["cluster"], agg["rel"], agg["gender"], agg["replace"],
                            agg["dates"], agg["ages"], agg["locations"])
    print("\n" + "=" * 74)
    print(f"AGGREGATE (micro-averaged over {n if n is not None else '?'} transcripts)\n")
    print(f"  clustering recall : {_fmt(_acc(c['exact'], c['gold']))}  "
          f"(exact {c['exact']}/{c['gold']}, over-merges {c['over_merges']}, "
          f"splits {c['splits']}, coref-merges {c['coref_merges']})")
    print(f"  relation precision: {_fmt(_acc(r['tp'], r['pred']))}   "
          f"recall: {_fmt(_acc(r['tp'], r['gold']))}   "
          f"detail acc: {_fmt(_acc(r['detail_ok'], r['tp']))}")
    if r.get("llm_pred"):
        print(f"  llm relation path : {r['llm_pred']} edges -> +{r['llm_tp']} correct, "
              f"{r['llm_fp']} false-pos  (rules alone: {r['rule_tp']} correct, "
              f"{r['rule_fp']} false-pos)")
    print(f"  gender recall     : {_fmt(_acc(g['correct'], g['total']))}  "
          f"(wrong {g['wrong']}, missing {g['missing']})")
    print(f"  replace accuracy  : {_fmt(_acc(rp['correct'], rp['total']))}  "
          f"(LEAKS {rp['leaks']}, over-redactions {rp['over_redactions']})")
    print(f"  date accuracy     : {_fmt(_acc(d['pass'], d['total']))}  ({d['pass']}/{d['total']})")
    print(f"  age accuracy      : {_fmt(_acc(a['pass'], a['total']))}  ({a['pass']}/{a['total']})")
    print(f"  location typing   : {_fmt(_acc(l['typed'], l['in_gaz']))}  ({l['typed']}/{l['in_gaz']})"
          f"   off-gazetteer: {l['typed_off_gaz']}/{l['off_gaz']}"
          f"   LOCATED_IN: {l['edges']} ({l['edges_llm']} llm-verified)")
    if l.get("rep_total"):
        print(f"  place redaction   : {_fmt(_acc(l['rep_correct'], l['rep_total']))}  "
              f"(LEAKS {l['rep_leaks']}, over-redactions {l['rep_over']})")
    for label, key in (("date redaction   ", "date_redaction"),
                       ("age redaction    ", "age_redaction")):
        b = agg.get(key) or {}
        if b.get("total"):
            print(f"  {label} : {_fmt(_acc(b['correct'], b['total']))}  "
                  f"({b['correct']}/{b['total']}, LEAKS {b['leaks']}, "
                  f"over-redactions {b['over']})")
    idg = agg.get("identifying") or {}
    if idg.get("total"):
        print(f"  identifying (occ) : {_fmt(_acc(idg['correct'], idg['total']))}  "
              f"({idg['correct']}/{idg['total']}, wrong {idg['wrong']})")
    o = agg.get("owner") or {}
    if o.get("total"):
        print(f"  ownership accuracy: {_fmt(_acc(o['correct'], o['total']))}  "
              f"(correct {o['correct']}/{o['total']}, WRONG {o['wrong']}, "
              f"unresolved {o['missing']})")
    iv = agg.get("interviewee") or {}
    if iv:
        parts = []
        for f in ("identity", "gender", "ethnicity", "dob"):
            counts = iv.get(f)
            if not counts:
                continue
            good = counts.get("correct", 0)
            tot = sum(v for k, v in counts.items() if k != "n/a")
            bad = counts.get("wrong", 0)
            parts.append(f"{f} {good}/{tot}" + (f" (WRONG {bad})" if bad else ""))
        print(f"  INTERVIEWEE       : " + "   ".join(parts))

    s = agg["second_line"]
    print("\n  SECOND LINE (graph.second_line.resolve_all)"
          f"  llm_ran={s['llm_ran']}" + (f" ({s['llm_model']})" if s.get("llm_model") else ""))
    tot = sum(s["actions"].values())
    order = ("confirm", "fill", "keep", "conflict", "refuted", "blind")
    parts = [f"{k} {s['actions'].get(k, 0)}" for k in order if s["actions"].get(k)]
    print(f"    {tot} field resolutions: " + ", ".join(parts))
    print(f"    blocking fields: {s['blocking']}")
    for fname in sorted(s["by_field"]):
        counts = s["by_field"][fname]
        line = ", ".join(f"{k} {counts[k]}" for k in order if counts.get(k))
        print(f"      {fname:<18s} {line}")

    print("\n  Reminder: assumes a perfect detector -> this is an upper bound on")
    print("  your stage. Multiply by the detector's recall for end-to-end numbers.")
    print("  For interviewee-only surrogate generation the rows that matter are")
    print("  INTERVIEWEE, ownership accuracy and place redaction -- a WRONG owner or a")
    print("  leaked place is unrecoverable; an unresolved one blocks and reaches a human.")


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
