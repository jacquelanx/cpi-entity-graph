"""
Evaluation harness for the knowledge-graph stage.

PURPOSE
    Answer "how good is this stage?" with numbers: precision / recall / accuracy
    per transcript and in aggregate, broken out by what the pipeline decides
    (clustering, relations, person subtype, gender, redaction, dates, ages, place
    typing, the place hierarchy, ownership, the interviewee's own fields).

READING THE NUMBERS
    Three habits here exist to stop a percentage from flattering the pipeline:

      * a gold row the pipeline built NO ENTITY for counts as a FAILURE, in the
        direction its field makes it dangerous. Skipping those rows removed them
        from the numerator and the denominator both;
      * every boolean metric prints its gold CLASS BALANCE. An all-True or
        all-False gold set is satisfied by a constant answer, so the percentage is
        a regression guard and not an accuracy -- `identifying` is marked exactly
        that way;
      * values are compared to gold, not tested for presence. Place typing used to
        count any non-empty subtype as correct and report 100% while a fifth of the
        types were wrong.

    And two that stop it from flattering the HARNESS: an ancestor edge is credited
    as a coarse hit rather than punished as both a miss and a false positive, and
    where gold asserts nothing (`Tug Fork` has no agreed container) the pipeline's
    answer is left unscored rather than counted against it.

FIT
    A CONSUMER of `graph/`, not part of it -- `scripts/eval.py` is the entry
    point, and `graph.pipeline.run_pipeline` is called through the same front door
    the demos and reports use. Reads the hand-written gold annotations in
    `samples/gold/`.

HOW
    Simulates a PERFECT DETECTOR: instead of running a real detector, the gold
    surface forms are located in the transcript and turned into detection spans.
    That isolates this stage -- a miss in the numbers below is this stage's miss,
    not the detector's -- at the cost of making every figure an UPPER BOUND.

It simulates a perfect detector from the gold annotations in `samples/gold/` and
runs those spans through `graph.pipeline.run_pipeline` -- THE SAME entry point
the demos and the reports use -- so a value the second line CONFIRMED, FILLED
after its checks, or REJECTED is scored exactly as a consumer would see it.

Relations are broken down by provenance (rule vs llm) so the LLM path's recall
gain and precision cost are both visible.

    ./venv/bin/python3 scripts/eval.py        # or: python3 -m evaluation.cli

  KG_USE_LLM=1     score rules+LLM (needs Ollama); unset = rules-only baseline
  EVAL_NO_COREF=1  skip the coref stage

    config.py        env flags, paths, the shared LLM client. IMPORT FIRST.
    kin_synonyms.py  kin-word synonyms -> canonical family term
    loc_buckets.py   place-type words -> canonical bucket
    metrics.py       the accuracy ratio
    detections.py    the simulated perfect detector
    scoring.py       `evaluate_one` -- score one transcript
    report.py        print per-transcript and aggregate blocks
    cli.py           `main`

`kin_synonyms.py` and `loc_buckets.py` are deliberate parallels to
`graph/checks/comparators._KIN_CANON` and `graph/checks/location.LOC_CANON`, kept
separate so the harness never grades the pipeline using the pipeline's own notion
of agreement -- widening a vocabulary in `graph/` must not silently widen what
counts as a right answer here.

Reminder: a perfect detector makes every number here an UPPER BOUND on this
stage. Multiply by the detector's recall for end-to-end figures.
"""
