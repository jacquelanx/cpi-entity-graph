NOTE TO SELF (COMMANDS):
Activate virtual environment: source venv/bin/activate
Run eval test script on pipeline: ./venv/bin/python3 scripts/eval.py
Generate HTML dashboard: ./venv/bin/python3 scripts/pipeline_report.py

---

## Brief summary of this codebase

**What this codebase does.** This codebase takes detected people/places/dates/etc
in a certain format (VERY IMPORTANT; see below for the precise format) and turn them
into an entity knowledge graph. Basically, this code clusters every mention of the 
same person together, pull out relationships ("my aunt Maria" → interviewee–aunt→Maria)
so they can be maintained during surrogate generation, keeps track of dates and ages
and location hierarchies (again, so surrogate generation is consistent later), and infer
attributes (gender, people's profession). All of this is so that we are able to easily
do consistent, accurate surrogate generation later. 


**If you want to look at the output of this stage, run this command to see a HTML report:** 
`./venv/bin/python3 scripts/pipeline_report.py`--it opens an HTML report walking through every stage 
on 5 sample transcripts I generated, with precision/recall/accuracy numbers per transcript.
These 5 sample transcripts can be found in the `samples/transcripts/` folder. Feel free to use
these 5 transcripts to test detector outputs.
NOTE: disregard what's in `samples/gold/`!! Those json files are for my reference only and are
NOT the output that the entity graph ingests. See the next bullet point for the actual format
that this codebase ingests currently.


**IMPORTANT: the output format from the detection stage.** One JSON file per transcript,
exactly in this format:

```json
{
  "transcript_id": "interview_001",
  "detector_version": "0.3",
  "detections": [
    { 
      "start": 42, 
      "end": 52, 
      "entity_type": "PERSON",
      "score": 0.9, 
      "text": "Aunt Maria",  
      "recognizer": "spacy" 
    }
  ]
}
```
Where "entity_type" can only be from the categories we defined, restated here:
"PERSON", "NICKNAME", "LOCATION", "DATE_ABSOLUTE", "DATE_RELATIVE", "DATE_ANCHOR",
"AGE", "PHONE", "EMAIL", "SSN_OR_ID", "USERNAME_HANDLE", "DATE_OF_BIRTH", "INSTITUTION", 
"OCCUPATION".


Also note: **`start`/`end` are character offsets into the raw `.txt`, 0-indexed and 
end-exclusive** — i.e. `transcript[start:end]` has to equal `text` character-for-character. 
- `score` and `recognizer` are optional but nice to have.
- The DATE split is important for this stage: `DATE_ABSOLUTE` (parseable, "March 2020"),
  `DATE_RELATIVE` ("two years ago"), `DATE_ANCHOR` (public events like "Hurricane Katrina"
  or "9/11"). 
- Overlapping spans are fine in the detection stage!

---

## Repo layout

```
graph/                the deterministic stage. Top level is the spine:
  models.py             core dataclasses (Mention / Entity / Relation / Edge)
  loader.py             validates detector JSON -> Mentions            (input boundary)
  pipeline.py           orchestrates every stage                       (run_pipeline)
  second_line/          THE arbitration point: rule vs LLM, per field
    outcomes.py           outcomes, tiers, `Resolution`, `FieldPolicy` (pure data)
    engine.py             `second_line` -- decide ONE field
    safe_direction.py     the SAFE_DIRECTION conflict resolvers
    policies.py           THE registry: one `FieldPolicy` per field
    apply.py              write a decision back onto an entity
    walk.py               `resolve_all` -- drive all of it over a transcript
  serialize.py          builds + validates the artifact                (output boundary)
  text/                 transcript utilities with no graph knowledge
    sentences.py          abbreviation-aware sentence spans
    turns.py              speaker-turn segmentation / subject masking
  rules/                the rule layer -- one module per inference stage
    name_matching.py      clustering part 1: normalize + merge by name
    aliases.py            explicit "we called her Glo" alias cues
    coref.py              clustering part 2: fastcoref + LLM double-gate
    kinship.py            clustering part 3: kin relation edges
    interviewee.py        which named PERSON is the speaker
    attributes.py         gender / given_name / surname / role / ethnicity
    identifiers.py        PHONE / EMAIL / SSN_OR_ID / USERNAME_HANDLE / OCCUPATION
    locations.py          gazetteer lookup + LOCATED_IN hierarchy
    dates.py              absolute / relative / anchor date resolution
    ages.py               age parsing + the age <-> date constraint
  checks/               deterministic checkers -- one module per FIELD checked
    relation_evidence.py  the transcript-evidence verifier behind checks/relations.py

llm_layer/            optional local-LLM PROPOSERS (graph arbitrates; one-way dep)
  client.py             Ollama client + persistent cache
  extract.py            windowed read-along pass
  openworld.py          per-entity open-world proposer
  identifier_judge.py   windowed identifier judgment
  interviewee.py        proposes which named person is the speaker
  merge_adjudicate.py   "are these two mentions the same person?"

demo/                 libraries behind the sample-transcript reports
  cases.py              simulated perfect detector -> run_pipeline
  render/               HTML fragments shared by both report pages
    primitives.py         formatting helpers, palette + label tables
    provenance.py         action badges, check lists, review flags
    stages_cluster.py     detect -> cluster -> coref -> relations
    stages_people.py      interviewee identification + person cards
    stages_world.py       places, dates/ages, identifiers
    stages_graph.py       graph SVG, edge table, ledger, artifact
    page.py               metrics tiles, stepper, `transcript_panel`
    css.py                the stylesheet
evaluation/           scoring harness (precision / recall / accuracy)
  config.py             env flags, paths, LLM client. IMPORT FIRST.
  kin_synonyms.py       kin-word synonyms -> canonical family term
  metrics.py            accuracy ratio + no-entity placeholder
  detections.py         the simulated perfect detector
  scoring.py            `evaluate_one` -- score one transcript
  report.py             per-transcript + aggregate console blocks
  cli.py                `main`
scripts/              runnable entry points ONLY
  build_graph.py        transcript + detections -> out/graphs/<id>.json
  eval.py               -> evaluation.cli
  pipeline_report.py    -> reports/pipeline_report.html  (rules only)
  llm_report.py         -> reports/llm_report.html       (rules + LLM)
samples/              sample transcripts + gold annotations (fixtures)
reports/              generated HTML reports
tests/                test_invariants.py -- deterministic regression guards
data/                 gazetteer.csv + metadata.json (not committed)
out/                  generated artifacts (not committed)
```

Module and package names are `lower_snake_case` noun phrases with every word
boundary marked by `_`. Inside `graph/checks/`, a module is named for the field
it checks (`ages.py`, `dates.py`, `stated_with.py`, `relations.py`).
