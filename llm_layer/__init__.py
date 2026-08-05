"""
Optional local-LLM layer for the de-identification pipeline. Sits on top of the
deterministic ruleset in `graph/` and adds a second, conservative judgment.

Design invariant -- the LLM is a SECOND LINE OF DEFENSE, never authoritative:
every rule table/parser can miss, so wherever a rule leaves a value unset,
unresolved, ambiguous, or malformed, one of these passes steps in with a FLAGGED
SUGGESTION (`suggested_*` / `candidate_*` + a review flag). The LLM never
overwrites a rule-set value and never lowers redaction. Everything detected in a
transcript now has such a fallback:

  clustering (ambiguous names) .. adjudicate_same_person   (pipeline._adjudicate_ambiguous)
  relations, known vocabulary ... extract_pass -> relation_verify (apply/suggest)
  relations, unknown vocabulary . relation_verify           (suggest, tagged raw word)
  gender / role / ethnicity ..... extract_pass               (suggested_gender / _role / _ethnicity)
  public figure & redaction ..... openworld_pass             (co-sign keep / candidate / raise)
  FAMILY / PROFESSIONAL subtype . openworld_pass             (suggested_subtype)
  location type / hierarchy ..... openworld_pass             (suggested_type / _parent)
  anchor / absolute / relative dates .. openworld_pass       (check + suggested_value)
  age value ..................... openworld_pass             (check + suggested_value)
  identifier owner / identifying  identifier_judge_pass      (owner / identifying)
  identifier type (malformed) ... identifier_judge_pass      (suggested_kind)

Public API:
  default_client / LLMClient   -- shared Ollama client + persistent cache (llm.py)
  adjudicate_same_person       -- use #1: merge adjudication  (merge_adjudicate.py)
                                   also the clustering-ambiguity second line
  openworld_pass               -- use #2: open-world classifier (openworld.py)
                                   (public figure / subtype / location / date / age)
  extract_pass                 -- use #3: windowed extraction    (extract.py)
                                   (attributes + relations + aliases)
  identifier_judge_pass        -- use #4: identifier judgment (identifier_judge.py)
                                   (owner / occupation identifying-ness / type)
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
