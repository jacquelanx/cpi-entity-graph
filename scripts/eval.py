"""
Entry point for the evaluation harness. The implementation lives in
`evaluation/` -- see that package's docstring for what is scored and why.

    ./venv/bin/python3 scripts/eval.py

  KG_USE_LLM=1     score rules+LLM (needs Ollama); unset = rules-only baseline
  EVAL_NO_COREF=1  skip the coref stage
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from evaluation.cli import main

if __name__ == "__main__":
    main()
