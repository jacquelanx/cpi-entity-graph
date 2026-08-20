"""
Evaluation harness for the knowledge-graph stage.

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
