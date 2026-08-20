"""
`graph.rules` -- the RULE layer, one module per inference stage.

Package marker. Every module here computes a value deterministically from the
transcript, using closed vocabularies and regexes rather than a model, and
abstains rather than guessing. `graph/pipeline.py` calls them in order;
`graph/checks/` verifies what they (and the LLM) produce.

    name_matching.py  clustering part 1: normalize + merge by name
    aliases.py        explicit "we called her Glo" alias cues
    coref.py          clustering part 2: fastcoref + LLM double-gate
    kinship.py        RELATED_TO edges from family words
    interviewee.py    which named PERSON is the speaker
    attributes.py     gender / name parts / role / ethnicity
    identifiers.py    PHONE / EMAIL / SSN_OR_ID / USERNAME_HANDLE / OCCUPATION
    locations.py      gazetteer lookup + the LOCATED_IN hierarchy
    dates.py          absolute / relative / anchor date resolution
    ages.py           age parsing + the age <-> date constraint
"""
