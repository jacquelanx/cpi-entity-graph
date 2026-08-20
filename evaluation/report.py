"""
Printing the scores to the console: one block per transcript, then the aggregate.

PURPOSE
    Turn the result dicts `scoring.evaluate_one` returns into readable console
    output, and accumulate them into the totals printed at the end.

FIT
    Called by `evaluation/cli.py`. Reads only `metrics._acc`, so it depends on
    nothing that computes a score.

HOW
    `_print_one` formats one transcript, `_accumulate` folds a result into the
    running totals, and `_print_aggregate` prints those totals. Percentages go
    through `_fmt`, which renders an unmeasurable value ("no cases") as "n/a"
    rather than as 0%.
"""

from __future__ import annotations

from collections import Counter
from .metrics import _acc


def _fmt(x):
    """Format a 0..1 ratio as a fixed-width percentage; None becomes "n/a".

    Fixed width so the columns line up when several transcripts are printed in
    sequence.
    """
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def _bal(b):
    """The "[gold all-X: no signal]" marker for a single-class boolean metric.

    A metric whose gold rows are all True (or all False) is satisfied by a constant
    answer, so its percentage measures nothing about the pipeline. Rather than drop
    the row -- it is still a useful regression guard -- the balance is printed
    beside it, so nobody reads the number as a result. Returns "" when both classes
    are present, which is the case worth a percentage.
    """
    pos, neg = b.get("gold_pos"), b.get("gold_neg")
    if pos is None or neg is None or (pos and neg):
        return ""
    return f"  [gold all-{'True' if pos else 'False'}: constant answer scores 100%]"


def _print_one(R):
    """Print the full score block for one transcript.

    `R` is a result dict from `scoring.evaluate_one`; the tuple unpacking on the
    next line just gives its seven sections short local names. Detail lines
    (misses, false positives, individual date/age failures) are printed only when
    non-empty, so a clean transcript stays compact.
    """
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
    # The per-person fields all print the same shape: correct / WRONG / missing,
    # because a wrong value misleads the next stage and a missing one only slows
    # it down. `alias` is the `same_person` positive direction -- its negative is
    # the over-merge count on the clustering line above.
    for label, key in (("subtype    ", "person_subtype"), ("role       ", "role"),
                       ("given name ", "given_name"), ("surname    ", "surname"),
                       ("ethnicity  ", "ethnicity")):
        b = R.get(key) or {}
        if not b.get("total"):
            continue
        extra = (f", unconfirmed {b['unconfirmed']}"
                 if b.get("unconfirmed") else "")
        print(f"  {label}: acc {_fmt(b['accuracy'])}  "
              f"(correct {b['correct']}/{b['total']}, WRONG {b['wrong']}, "
              f"missing {b['missing']}{extra})")
        for who, got, want in b["fail"]:
            print(f"      fail      : {who!r} -> {got} (want {want})")
    al = R.get("alias") or {}
    if al.get("total"):
        print(f"  alias merge: acc {_fmt(al['accuracy'])}  "
              f"({al['merged']}/{al['total']} multi-form people in one entity)")
        for who, forms in al["fail"]:
            print(f"      fail      : {who!r} split across entities {forms}")
    print(f"  replace    : acc {_fmt(rp['accuracy'])}  "
          f"(LEAKS {rp['leaks']}, over-redactions {rp['over_redactions']})"
          + _bal(rp))
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
    # The four date-shifter inputs. `stated_with` prints its class balance because
    # most ages have no anchor, so the null rows carry the precision signal.
    for label, key in (("shiftable  ", "shiftable"), ("approximate", "approximate"),
                       ("stated_with", "stated_with"), ("id kind    ", "kind")):
        b = R.get(key) or {}
        if not b.get("total"):
            continue
        print(f"  {label}: acc {_fmt(b['accuracy'])}  "
              f"(correct {b['correct']}/{b['total']}, WRONG {b['wrong']}, "
              f"missing {b['missing']})"
              + (f"  [gold {b['gold_pos']} pos / {b['gold_neg']} neg]"
                 if key == "stated_with" else
                 "  [detector supplies it: regression guard]" if key == "kind"
                 else _bal(b)))
        for txt, got, want in b["fail"]:
            print(f"      fail      : {txt!r} -> {got} (want {want})")
    # TYPE CORRECTNESS, not type presence -- `type_ok` compares the pipeline's
    # subtype to the gold type through the harness's own bucket table, where the old
    # number only asked whether a subtype existed at all.
    print(f"  loc typing : acc {_fmt(l['accuracy'])}  "
          f"({l['type_ok']}/{l['in_gaz']} in gazetteer, typed {l['typed']}, "
          f"WRONG {l['type_wrong']})   off-gazetteer {_fmt(l['off_accuracy'])} "
          f"({l['off_type_ok']}/{l['off_gaz']}, typed {l['typed_off_gaz']}, "
          f"WRONG {l['off_type_wrong']})")
    for txt, got, want in l.get("type_fail", []):
        print(f"      fail      : {txt!r} -> {got} (want {want})")
    lp = R.get("loc_parent") or {}
    if lp.get("gold") or lp.get("pred"):
        print(f"  loc parent : P {_fmt(lp['precision'])}  R {_fmt(lp['recall'])}  "
              f"(tp {lp['tp']}/{lp['scorable']} scorable of {lp['gold']} gold, "
              f"exact {lp['exact']}, pred {lp['pred']}, {l['edges_llm']} llm-verified)")
        if lp["misses"]:
            print(f"      misses    : {lp['misses']}")
        if lp["false_pos"]:
            print(f"      false pos : {lp['false_pos']}")
        if lp["coarse"]:
            print(f"      coarse    : {lp['coarse']}")
        if lp["unscored_edges"]:
            print(f"      unscored  : {lp['unscored_edges']}  (no gold parent asserted)")
    if l["rep_total"]:
        print(f"  place redac: acc {_fmt(l['rep_accuracy'])}  "
              f"(LEAKS {l['rep_leaks']}, over-redactions {l['rep_over']})" + _bal(l))
    for label, key in (("date redac ", "date_redaction"), ("age redac  ", "age_redaction")):
        b = R[key]
        if not b["total"]:
            continue
        print(f"  {label}: acc {_fmt(b['accuracy'])}  ({b['correct']}/{b['total']}, "
              f"LEAKS {b['leaks']}, over-redactions {b['over']})" + _bal(b))
        for txt, got, want in b["fail"]:
            print(f"      fail      : {txt!r} -> {got} (want {want})")
    idg = R["identifying"]
    if idg["total"]:
        print(f"  identifying: acc {_fmt(idg['accuracy'])}  "
              f"({idg['correct']}/{idg['total']} occupations, wrong {idg['wrong']})"
              + _bal(idg))
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
        if "given_name" in iv:
            print(f"             : given {iv['given_name']} ({v['given_name']})  "
                  f"surname {iv['surname']} ({v['surname']})")

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
    """Fold one transcript's result into the running aggregate, in place.

    Three different accumulation strategies, because the sections are different
    shapes:

      COUNTS      the seven scored sections plus owner / redaction / identifying are
                  micro-averaged: every numeric field is SUMMED, and the ratios are
                  recomputed at the end from those sums. `isinstance(v, bool)` is
                  excluded because `bool` is a subclass of `int` in Python, so a
                  True flag would otherwise be summed as 1.
      VERDICTS    the interviewee block is one verdict per field per transcript, so
                  it counts outcomes in a `Counter` rather than summing numbers.
      SECOND LINE action counts are unioned per field, and `llm_ran` is OR-ed, so
                  the aggregate says "the LLM ran somewhere" if it ran anywhere.
    """
    for k in ("cluster", "rel", "gender", "replace", "dates", "ages", "locations",
              "loc_parent", "person_subtype", "role", "given_name", "surname",
              "ethnicity", "alias", "shiftable", "approximate", "stated_with",
              "kind",
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
    """Print the totals across all transcripts.

    MICRO-averaged: every ratio is recomputed from the summed numerators and
    denominators, so a transcript with more entities contributes proportionally
    more. (A macro average -- the mean of the per-transcript percentages -- would
    let a transcript with two dates weigh as much as one with twenty.)
    """
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
    for label, key in (("subtype accuracy  ", "person_subtype"),
                       ("role accuracy     ", "role"),
                       ("given-name acc    ", "given_name"),
                       ("surname accuracy  ", "surname"),
                       ("ethnicity (others)", "ethnicity")):
        b = agg.get(key) or {}
        if not b.get("total"):
            continue
        extra = f", unconfirmed {b['unconfirmed']}" if b.get("unconfirmed") else ""
        print(f"  {label}: {_fmt(_acc(b['correct'], b['total']))}  "
              f"({b['correct']}/{b['total']}, WRONG {b['wrong']}, "
              f"missing {b['missing']}{extra})")
    al = agg.get("alias") or {}
    if al.get("total"):
        print(f"  alias merge       : {_fmt(_acc(al['merged'], al['total']))}  "
              f"({al['merged']}/{al['total']} multi-form people in one entity)")
    print(f"  replace accuracy  : {_fmt(_acc(rp['correct'], rp['total']))}  "
          f"(LEAKS {rp['leaks']}, over-redactions {rp['over_redactions']})" + _bal(rp))
    print(f"  date accuracy     : {_fmt(_acc(d['pass'], d['total']))}  ({d['pass']}/{d['total']})")
    print(f"  age accuracy      : {_fmt(_acc(a['pass'], a['total']))}  ({a['pass']}/{a['total']})")
    # The date-shifter's four inputs, printed together: they are consumed together.
    for label, key in (("shiftable         ", "shiftable"),
                       ("approximate       ", "approximate"),
                       ("stated_with (age) ", "stated_with"),
                       ("identifier kind   ", "kind")):
        b = agg.get(key) or {}
        if not b.get("total"):
            continue
        print(f"  {label}: {_fmt(_acc(b['correct'], b['total']))}  "
              f"({b['correct']}/{b['total']}, WRONG {b['wrong']}, "
              f"missing {b['missing']})"
              + (f"  [gold {b['gold_pos']} pos / {b['gold_neg']} neg]"
                 if key == "stated_with" else
                 "  [detector supplies it: regression guard]" if key == "kind"
                 else _bal(b)))
    print(f"  location typing   : {_fmt(_acc(l['type_ok'], l['in_gaz']))}  "
          f"({l['type_ok']}/{l['in_gaz']} in gazetteer, WRONG {l['type_wrong']})"
          f"   off-gazetteer: {_fmt(_acc(l['off_type_ok'], l['off_gaz']))} "
          f"({l['off_type_ok']}/{l['off_gaz']}, WRONG {l['off_type_wrong']})")
    lp = agg.get("loc_parent") or {}
    if lp.get("gold") or lp.get("pred"):
        print(f"  location hierarchy: P {_fmt(_acc(lp['tp'], lp['scored_pred']))}  "
              f"R {_fmt(_acc(lp['tp'], lp['scorable']))}  "
              f"(tp {lp['tp']}/{lp['scorable']} scorable of {lp['gold']} gold parents, "
              f"exact {lp['exact']}, FP {lp['fp']}, {lp['unscored']} unscored, "
              f"{l['edges_llm']} llm-verified)")
    if l.get("rep_total"):
        print(f"  place redaction   : {_fmt(_acc(l['rep_correct'], l['rep_total']))}  "
              f"(LEAKS {l['rep_leaks']}, over-redactions {l['rep_over']})" + _bal(l))
    for label, key in (("date redaction   ", "date_redaction"),
                       ("age redaction    ", "age_redaction")):
        b = agg.get(key) or {}
        if b.get("total"):
            print(f"  {label} : {_fmt(_acc(b['correct'], b['total']))}  "
                  f"({b['correct']}/{b['total']}, LEAKS {b['leaks']}, "
                  f"over-redactions {b['over']})" + _bal(b))
    idg = agg.get("identifying") or {}
    if idg.get("total"):
        print(f"  identifying (occ) : {_fmt(_acc(idg['correct'], idg['total']))}  "
              f"({idg['correct']}/{idg['total']}, wrong {idg['wrong']})" + _bal(idg))
    o = agg.get("owner") or {}
    if o.get("total"):
        print(f"  ownership accuracy: {_fmt(_acc(o['correct'], o['total']))}  "
              f"(correct {o['correct']}/{o['total']}, WRONG {o['wrong']}, "
              f"unresolved {o['missing']})")
    iv = agg.get("interviewee") or {}
    if iv:
        parts = []
        for f in ("identity", "given_name", "surname", "gender", "ethnicity", "dob"):
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
