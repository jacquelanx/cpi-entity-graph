"""
Optional local-LLM layer for the de-identification pipeline. Sits on top of the
deterministic ruleset in `graph/` and adds a second, conservative judgment. 
Public API:
  default_client / LLMClient   -- shared Ollama client + persistent cache (llm.py)
  adjudicate_same_person       -- use #1: merge adjudication  (merge_adjudicate.py)
  openworld_pass               -- use #2: open-world classifier (openworld.py)
  extract_pass                 -- use #3: windowed extraction    (extract.py)
                                   (attributes + relations + aliases)
  identifier_judge_pass        -- use #4: identifier judgment (identifier_judge.py)
                                   (owner / occupation identifying-ness)
"""

from .llm import LLMClient, default_client
from .merge_adjudicate import adjudicate_same_person
from .openworld import openworld_pass
from .extract import extract_pass
from .identifier_judge import identifier_judge_pass

__all__ = [
    "LLMClient", "default_client", "adjudicate_same_person",
    "openworld_pass", "extract_pass", "identifier_judge_pass",
]
