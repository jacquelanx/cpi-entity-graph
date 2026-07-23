NOTE TO SELF (COMMANDS):
Activate virtual environment: source venv/bin/activate
Run eval test script on pipeline: ./venv/bin/python3 scripts/eval.py
Generate HTML dashboard: ./venv/bin/python3 scripts/dashboard.py

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
`./venv/bin/python3 scripts/dashboard.py`--it opens an HTML report walking through every stage 
on 5 sample transcripts I generated, with precision/recall/accuracy numbers per transcript.
These 5 sample transcripts can be found in the `tests/transcripts/` folder. Feel free to use
these 5 transcripts to test detector outputs.
NOTE: disregard what's in `tests/gold/`!! Those json files are for my reference only and are
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