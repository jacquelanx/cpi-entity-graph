"""
Evaluation script for the knowledge graph part ONLY. The knowledge graph consumes
detected spans from the detection stage; this file simulates a perfect detector and
runs those spans through the knowledge graph pipeline.

Relations are scored across BOTH layers: the deterministic ruleset always, and --
when the LLM is enabled (KG_USE_LLM=1 with Ollama up) -- the verified LLM relation
path (llm_layer.extract_pass -> relation_verify), added the same way the real
pipeline adds it. The relation report breaks results down by provenance (rule vs
llm) so the LLM's recall gain and its precision cost are both visible, and a
regression in the LLM path is caught here instead of only in the demo.
"""

from __future__ import annotations
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

# quiet fastcoref / transformers / datasets chatter before importing them
for _n in ("fastcoref", "transformers", "datasets", "urllib3"):
    logging.getLogger(_n).setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from dateutil import parser as dateparser
from graph.loader import resolve_overlaps, make_mentions
from graph.merge_strings import merge_person_mentions
from graph.aliases import apply_alias_cues
from graph.coref import apply_coref
from graph.kinship import extract_kinship
from graph.attributes import infer_person_attributes
from graph.location_dates import (
    load_gazetteer, build_location_edges, resolve_date_entity, resolve_age_entity,
)
from graph.models import Entity, Mention, Relation, Edge

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
    "grandmother": "grandmother grandma grandmom granny nana nanna gramma grammy meemaw",
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
    for loc in gold.get("locations", []):
        add(loc["text"], "LOCATION")
    for d in gold.get("dates", []):
        add(d["text"], d["category"])
    for a in gold.get("ages", []):
        add(a["text"], "AGE")
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

    # ---- run the pipeline on simulated-perfect detections ----
    dets = resolve_overlaps(_build_detections(text, gold))
    mentions = make_mentions(tid, dets)

    persons, _ambig = merge_person_mentions(tid, mentions, text, _LLM)
    # rule-based alias/nickname merges (closed cue set), independent of coref
    apply_alias_cues(text, persons)

    # Coreference stage (part 2 of clustering). Runs by default; folds coref
    # clusters into our person entities and can merge split name forms.
    coref_merges = 0
    if RUN_COREF and persons:
        persons, merged_pairs, _ran = apply_coref(text, persons, llm=_LLM)
        coref_merges = len(merged_pairs)

    interviewee = Entity(entity_id=f"{tid}_e000", category="PERSON")
    edges = extract_kinship(text, persons, interviewee)
    infer_person_attributes(text, persons, edges)

    # LLM relation path (verified). Mirrors graph/pipeline: additive, only when a
    # (source, target) pair isn't already a rule edge, so rules stay authoritative.
    # No-ops when the LLM is unavailable -> the rules-only column is unaffected.
    if _LLM is not None and _LLM.available():
        from llm_layer import extract_pass
        llm_rels = extract_pass(text, persons, interviewee, _LLM)
        have = {(e.source, e.target) for e in edges if e.relation == Relation.RELATED_TO}
        for r in llm_rels:
            if (r["source"], r["target"]) not in have:
                edges.append(Edge(source=r["source"], target=r["target"],
                                  relation=Relation.RELATED_TO, detail=r["detail"],
                                  evidence=f"(llm) {r['evidence']}"))
                have.add((r["source"], r["target"]))

    by_id = {e.entity_id: e for e in persons}
    by_id[interviewee.entity_id] = interviewee

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
    iv = dateparser.parse(gold["interview_date"]).date()
    d_total = d_pass = 0
    d_fail = []
    for d in gold.get("dates", []):
        spans = _find_spans(text, d["text"])
        if not spans:
            continue
        s, e = spans[0]
        ent = Entity(entity_id="d", category=d["category"],
                     mentions=[Mention(tid, s, e, d["text"], d["category"], "d")])
        resolve_date_entity(ent, iv)
        got = ent.attributes.get("resolved_value")
        d_total += 1
        if d.get("expect_flag"):
            ok = got is None
        else:
            ok = got is not None and abs(
                (dateparser.parse(got).date()
                 - dateparser.parse(d["resolved"]).date()).days) <= d.get("tolerance_days", 0)
        d_pass += 1 if ok else 0
        if not ok:
            d_fail.append((d["text"], got, d.get("resolved") or "FLAG"))
    R["dates"] = {"total": d_total, "pass": d_pass,
                  "accuracy": _acc(d_pass, d_total), "fail": d_fail}

    # ---------- AGES ----------
    a_total = a_pass = 0
    a_fail = []
    for a in gold.get("ages", []):
        spans = _find_spans(text, a["text"])
        if not spans:
            continue
        s, e = spans[0]
        ent = Entity(entity_id="a", category="AGE",
                     mentions=[Mention(tid, s, e, a["text"], "AGE", "a")])
        resolve_age_entity(ent)
        got = ent.attributes.get("value")
        a_total += 1
        ok = got is not None and abs(got - a["value"]) <= a.get("tolerance", 0)
        a_pass += 1 if ok else 0
        if not ok:
            a_fail.append((a["text"], got, a["value"]))
    R["ages"] = {"total": a_total, "pass": a_pass,
                 "accuracy": _acc(a_pass, a_total), "fail": a_fail}

    # ---------- LOCATIONS (gazetteer typing) ----------
    records, aliases = load_gazetteer(DATA_GAZ)
    loc_mentions = [m for m in mentions if m.entity_type == "LOCATION"]
    loc_entities = [Entity(entity_id=f"{tid}_L{i}", category="LOCATION", mentions=[m])
                    for i, m in enumerate(loc_mentions)]
    build_location_edges(loc_entities, records, aliases)
    gaz_gold = [l for l in gold.get("locations", []) if l.get("in_gazetteer")]
    typed = 0
    for l in gaz_gold:
        if any(le.mentions[0].text == l["text"] and le.subtype for le in loc_entities):
            typed += 1
    R["locations"] = {"in_gaz": len(gaz_gold), "typed": typed,
                      "accuracy": _acc(typed, len(gaz_gold))}
    return R


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
    print(f"  dates      : acc {_fmt(d['accuracy'])}  ({d['pass']}/{d['total']})")
    if d["fail"]:
        print(f"      fails     : {d['fail']}")
    print(f"  ages       : acc {_fmt(a['accuracy'])}  ({a['pass']}/{a['total']})")
    if a["fail"]:
        print(f"      fails     : {a['fail']}")
    print(f"  locations  : typed {_fmt(l['accuracy'])}  ({l['typed']}/{l['in_gaz']} in gazetteer)")


def _accumulate(agg, R):
    for k in ("cluster", "rel", "gender", "replace", "dates", "ages", "locations"):
        bucket = agg.setdefault(k, {})
        for f, v in R[k].items():
            if isinstance(v, (int, float)):
                bucket[f] = bucket.get(f, 0) + v


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
    print(f"  location typing   : {_fmt(_acc(l['typed'], l['in_gaz']))}  ({l['typed']}/{l['in_gaz']})")
    print("\n  Reminder: assumes a perfect detector -> this is an upper bound on")
    print("  your stage. Multiply by the detector's recall for end-to-end numbers.")


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
