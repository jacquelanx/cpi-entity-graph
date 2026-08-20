"""
Optional local-LLM layer. It PROPOSES; `graph.second_line` ARBITRATES.

PURPOSE
    Ask a local language model the same questions the rule tables in `graph/` try
    to answer, for EVERY entity -- whether or not the rule found a value -- and
    return the answers as plain data. Nothing here decides anything.

FIT
    A leaf package with a strictly ONE-WAY dependency: `graph` imports
    `llm_layer`, never the reverse. That is enforced by the interface -- proposals
    are plain dicts of the shape below, so this package needs no knowledge of
    `Entity`, `Resolution` or the checkers. `graph/pipeline.run_pipeline` calls the
    three passes; `graph/rules/coref.py` and `graph/rules/name_matching.py` call
    the merge adjudicator directly.

HOW
    `client.py` holds one Ollama client with a persistent, privacy-preserving
    cache; the four task modules each build a prompt, call `client.judge`, and
    normalize the reply. Every call runs at temperature 0 with JSON-constrained
    output, so the same transcript yields the same proposals. If Ollama is not
    running the whole layer degrades to returning nothing and the pipeline is
    rules-only.

Every rule table and parser in `graph/` can miss, so this layer asks the model the
same questions the tables answer -- for every entity, whether or not the rule
filled the field -- and returns plain dicts:

    {entity_id: {field: {"value": ..., "confidence": ..., <extras>}}}

`graph.second_line.resolve_all` then decides, per field, with one of five
outcomes: a filled rule value is CHECKED (confirm / conflict), an empty one is
FILLED only if every deterministic checker in `graph/checks/` passes, and a field
neither layer could answer is an explicit REJECT rather than silence. Provenance
for each decision lands on `Entity.provenance`. Plain dicts keep the one-way
dependency (`graph` -> `llm_layer`) intact.

EVERY field is routed through the unified second line, and every field has a proposer.
Two holes used to hide in this list rather than in the code:

  * `approximate` was NOT proposed for DATE_ANCHOR (the anchor classifier returned no
    such key), so for any anchor phrase outside `ANCHOR_EVENTS` neither layer answered;
  * the windowed pass SKIPPED anyone the rules had typed PUBLIC_FIGURE, which removed
    the second line from `gender` / `given_name` / `surname` / `role` / `ethnicity` for
    exactly the entity that most needs it -- a private namesake of a celebrity, since
    the pass runs before `replace` is arbitrated.

Each row is `field .... proposer <- the rule table it second-lines`:

  interviewee identity .......... propose_interviewee <- interviewee.rule_candidate
  gender (named persons) ........ extract_pass      <- kinship.KINSHIP_GENDER
  gender (interviewee) .......... extract_pass      <- attributes.infer_interviewee_gender
  given_name / surname .......... extract_pass      <- attributes token split
  role .......................... extract_pass      <- attributes.infer_person_role
  ethnicity ..................... extract_pass      <- attributes.infer_ethnicity
  same_person (alias/nickname) .. extract_pass      <- aliases.apply_alias_cues + coref
  replace / PUBLIC_FIGURE ....... openworld_propose <- PUBLIC_FIGURES + _personal_signal
  FAMILY / PROFESSIONAL subtype . openworld_propose <- kin edges / PROFESSIONAL_CONTEXT
  location type ................. openworld_propose <- gazetteer type
  location parent (-> LOCATED_IN) openworld_propose <- gazetteer parent
  location replace .............. openworld_propose <- rules/locations.infer_location_replace
  date resolved_value ........... openworld_propose <- dateutil / rel-date regex /
                                                       season + spoken-year parsing /
                                                       ANCHOR_EVENTS
  shiftable (all date types) .... openworld_propose <- the rule's per-category default
  date replace (all types) ...... openworld_propose <- the resolved `shiftable`
  approximate (incl. ANCHORS) ... openworld_propose <- the rule's own marker
  age value ..................... openworld_propose <- word-number / decade maps
  age replace ................... openworld_propose <- checks/ages.age_reading_refuted
  age <-> date pairing .......... openworld_propose <- rules/ages.age_date_constraints
  identifier owner .............. identifier_judge_pass <- pipeline._link_interviewee_pii
  identifier kind + normalization identifier_judge_pass <- identifiers._normalize
  occupation identifying ........ identifier_judge_pass <- identifiers.COMMON_OCCUPATIONS

Merging is the one decision the LLM still cannot make on its own: a `same_person`
claim that clears every checker becomes a `suggested_merge_with` review flag, never
an automatic merge, because changing who the graph thinks exists is a human's call.

Public API:
  default_client / LLMClient   -- shared Ollama client + persistent cache (client.py)
  adjudicate_same_person      -- merge adjudication (merge_adjudicate.py)
  openworld_propose           -- per-entity proposer (openworld.py)
  extract_pass                -- windowed proposer + verified relations (extract.py)
  identifier_judge_pass       -- windowed identifier proposer (identifier_judge.py)
"""

from .client import LLMClient, default_client
from .merge_adjudicate import adjudicate_same_person
from .openworld import openworld_propose
from .extract import extract_pass
from .identifier_judge import identifier_judge_pass
from .interviewee import propose_interviewee

__all__ = [
    "LLMClient", "default_client", "adjudicate_same_person",
    "openworld_propose", "extract_pass", "identifier_judge_pass",
    "propose_interviewee",
]
