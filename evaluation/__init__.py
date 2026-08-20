"""
Evaluation harness for the knowledge-graph stage.

PURPOSE
    Answer "how good is this stage?" with numbers: precision / recall / accuracy
    per transcript and in aggregate, broken out by what the pipeline decides
    (clustering, relations, gender, redaction, dates, ages, places).

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
    metrics.py       accuracy ratio + the no-entity placeholder
    detections.py    the simulated perfect detector
    scoring.py       `evaluate_one` -- score one transcript
    report.py        print per-transcript and aggregate blocks
    cli.py           `main`

Reminder: a perfect detector makes every number here an UPPER BOUND on this
stage. Multiply by the detector's recall for end-to-end figures.
"""
