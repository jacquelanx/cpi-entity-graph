"""
Optional local-LLM layer for the de-identification pipeline. Sits on top of the
deterministic ruleset in `graph/` and adds a second, conservative judgment. 
Public API:
  default_client / LLMClient   -- shared Ollama client + persistent cache (llm.py)
  adjudicate_same_person       -- use #1: merge adjudication  (merge_adjudicate.py)
  openworld_pass               -- use #2: open-world classifier (openworld.py)
  infer_attributes_pass        -- use #3: attribute inference   (attr_infer.py)
"""

from .llm import LLMClient, default_client
from .merge_adjudicate import adjudicate_same_person
from .openworld import openworld_pass
from .attr_infer import infer_attributes_pass

__all__ = [
    "LLMClient", "default_client", "adjudicate_same_person",
    "openworld_pass", "infer_attributes_pass",
]
