"""
Run configuration for the evaluation harness: env flags, paths, the LLM client.

PURPOSE
    One place for everything the harness needs before it can run: which optional
    stages are on, where the samples live, and the shared LLM client.

FIT
    The bottom of `evaluation/` -- every other module in the package imports it,
    and the import ORDER matters (see below). Nothing here depends on `graph`
    except the lazy `llm_layer` client.

HOW -- and why it has to be imported first
    This module has IMPORT-TIME SIDE EFFECTS, and they only work if they happen
    before the libraries they configure are loaded: it puts the repo root on
    `sys.path`, silences the fastcoref / transformers / datasets loggers, and sets
    the HuggingFace verbosity environment variables. `graph.pipeline` pulls those
    libraries in transitively, so importing it first would leave the log noise in
    place.

IMPORT THIS FIRST from every other module in the package. Quieting the
fastcoref / transformers / datasets loggers only works if it happens BEFORE
those libraries are imported, and they are pulled in transitively by
`graph.pipeline`.

  EVAL_NO_COREF=1  skip the coref stage
  KG_USE_LLM=1     score rules+LLM instead of the rules-only baseline
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path


# quiet fastcoref / transformers / datasets chatter before importing them
for _n in ("fastcoref", "transformers", "datasets", "urllib3"):
    logging.getLogger(_n).setLevel(logging.ERROR)


os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


RUN_COREF = os.environ.get("EVAL_NO_COREF") != "1"


# opt-in local LLM adjudicator (KG_USE_LLM=1); no-ops if Ollama isn't running
_LLM = None


if os.environ.get("KG_USE_LLM") == "1":
    from llm_layer import default_client
    _LLM = default_client()


REPO = Path(__file__).resolve().parent.parent


ROOT = REPO / "samples"                     # transcripts + gold live here


DATA_GAZ = REPO / "data" / "gazetteer.csv"
