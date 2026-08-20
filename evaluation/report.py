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
