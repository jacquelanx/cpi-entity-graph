# Deterministic ruleset

Everything the graph pipeline decides via hand-written rules or the lookup table.

> **Two caveats.**
> 1. Every stage below is deterministic *except* the coreference model
>    (`fastcoref`) in stage 2. The model's prediction is statistical, but the
>    rules that decide whether to **accept, block, or flag** each of its
>    suggestions (stage 2b) are deterministic and documented here.
> 2. There is also an **optional local-LLM layer** (off by default) that adds a
>    second, conservative judgment on top of these rules. It changes the stage-2
>    merge behavior when enabled. 

Ruleset order: **load → cluster (rules) → coref (ML + rule guards) → kinship →
attributes → locations → dates → ages → age/date constraints**.

---

## 0. Input handling & validation (`graph/loader.py`)

- **Allowed `entity_type` labels** (anything else raises `Violation`):
  `PERSON`, `NICKNAME`, `LOCATION`, `DATE_ABSOLUTE`, `DATE_RELATIVE`,
  `DATE_ANCHOR`, `AGE`, `PHONE`, `EMAIL`, `SSN_OR_ID`, `USERNAME_HANDLE`,
  `DATE_OF_BIRTH`, `INSTITUTION`, `OCCUPATION`.
- **Offset validation:** each detection must satisfy
  `transcript[start:end] == text` exactly, or it raises `Violation`.
- **Overlap resolution rule** (`resolve_overlaps`): when two detected spans
  overlap, keep the **longer** one; ties broken by **higher `score`**. (So
  `"Aunt Maria"` wins over an inner `"Maria"`.)
- **Mention IDs:** assigned sequentially by start offset as
  `{transcript_id}_m0000`, `_m0001`, …

---

## 1. Rule-based person clustering (`graph/merge_strings.py`)

Groups `PERSON` / `NICKNAME` mentions into entities via a union-find. Only these
two label types are clustered here.

### 1a. Normalization applied before comparing names
- **Spell-out collapse:** `"H-A-Y-E-S"`, `"H A Y E S"`, `"H.A.Y.E.S"` → `"hayes"`
  (regex `^(?:[A-Za-z][\s\-.]){2,}[A-Za-z]\.?$`).
- **Lowercase**, strip surrounding `' " ( ) -` and `,`/`.`.
- **Strip-set removed from names** — `KINSHIP_AND_TITLES` (so `"Aunt Maria"` and
  `"Dr. Sarah Hayes"` normalize to `("maria",)` / `("sarah","hayes")`):
  - *Possessives:* my, his, her, their, our, your
  - *Titles:* mr, mr., mrs, mrs., ms, ms., miss, mx, dr, dr., prof, prof.,
    professor, sir, madam, ma'am, rev, rev., reverend, pastor, father, sister,
    brother, captain, capt, sergeant, sgt, officer, judge, senator, governor,
    gov, mayor, president, pres, coach, principal, auntie, uncle
  - *Kinship:* aunt, aunty, mother, mom, mommy, mum, mama, momma, ma, dad, daddy,
    papa, poppa, pop, pops, pa, grandma, grandpa, grandmother, grandfather,
    granny, nana, grandmom, grandad, granddad, gramps, cousin, sis, bro, husband,
    wife, son, daughter, niece, nephew, grandson, granddaughter, twin, partner,
    spouse, sibling, parent, child, kid, boyfriend, girlfriend, fiance, fiancee,
    stepmom, stepdad, stepmother, stepfather, stepsister, stepbrother, stepson,
    stepdaughter, godmother, godfather, brother-in-law, sister-in-law,
    mother-in-law, father-in-law, son-in-law, daughter-in-law, half-brother,
    half-sister
  - *Modifiers:* older, elder, younger, little, big, baby, oldest, youngest,
    eldest, only, middle, maternal, paternal, biological, bio, adoptive, adopted,
    foster, late, dear, beloved, step, half, great, grand, former, current, ex

### 1b. Merge rules (union-find)
1. **Exact normalized match** — same token tuple after normalization merges
   (e.g. `"Aunt Maria"` = `"Maria"` = `"maria"`). No nickname canonicalization.
2. **Token containment** — a single-token name merges into a multi-token name
   that contains it (`"Maria"` → `"Maria Rodriguez"`) **only if exactly one**
   such multi-token candidate exists; if two or more, it stays unmerged and is
   **flagged** `"short name matches multiple long names"`.
3. **Honorific collision guard** — for a **bare** (single-token) given name, the
   merge key also includes its honorific. So `"Miss Rosa"` and `"Rosa"` land in
   **different** entities (a titled + bare pair may be two people), and both are
   **flagged** `"bare given name also appears with a title elsewhere; possible
   distinct people"`. Kinship prefixes carry no honorific, so `"Aunt Maria"` and
   `"Maria"` still merge. Honorifics recognized (`HONORIFIC_TITLES`): mr, mrs, ms,
   miss, mx, dr, prof, professor, sir, madam, maam, ma'am, rev, reverend, pastor,
   captain, capt, sergeant, sgt, officer, judge, senator, governor, gov, mayor,
   president, pres, coach, principal.

---

## 2. Coreference (`graph/coref.py`)

`fastcoref` (ML) produces clusters of spans; we map each span to the entity it
overlaps. **2a is the model; 2b is the deterministic rule layer.** When the LLM
merge adjudicator is enabled, part of 2b defers to it.

### 2b. Accept/block/flag rules per coref-linked pair
Applied in order for two entities the model puts in one cluster:
1. **Block + flag if genders conflict** — if both have a non-null gender and they
   differ; **not merged**, both flagged. (Always a hard block.)
2. **Same-sentence co-occurrence** — if a mention of each sits in the same
   sentence (split on `. ! ?`):
   - **LLM off:** hard block + flag (treated as distinct people).
   - **LLM on:** *not* an automatic block — handed to the LLM, which reliably
     tells an explicit alias ("we called Roberto Beto") from two distinct people
     in one sentence ("Sarah's brother Danny").
3. **Merge decision:**
   - **LLM on:** the LLM must confirm same-person with high confidence or an
     evidence quote (the double-gate); otherwise flag, do not merge.
   - **LLM off:** merge only if **name-compatible** — share a token, or one token
     is a prefix of the other with both ≥ 3 chars (`"Will"`/`"William"`). 
4. **Otherwise flag, do not merge** — `"coref suggests same … but names differ;
   needs review"`.

(When a merge happens, mentions are combined, non-null attributes copied over, any
review flag on the absorbed entity is carried to the survivor, and an
`merge_evidence` attribute records the LLM's quote when present.)

---

## 3. Kinship relation extraction (`graph/kinship.py`)

Produces `RELATED_TO` edges and infers target gender from the kin word.

### 3a. Kin-word → gender table (`KINSHIP_GENDER`)
- **Female:** mother, mom, mommy, mum, mummy, mama, mamma, momma, ma, aunt,
  auntie, aunty, grandmother, grandma, grandmom, granny, nana, nanna, gramma,
  grammy, meemaw, sister, sis, wife, daughter, niece, granddaughter, stepmother,
  stepmom, stepsister, stepdaughter, half-sister, mother-in-law, sister-in-law,
  daughter-in-law, godmother, goddaughter, great-grandmother, fiancee, girlfriend,
  ex-wife
- **Male:** father, dad, daddy, papa, poppa, pop, pops, pa, uncle, grandfather,
  grandpa, granddad, grandad, grandpop, gramps, papaw, pawpaw, pappy, brother,
  bro, husband, son, nephew, grandson, stepfather, stepdad, stepbrother, stepson,
  half-brother, father-in-law, brother-in-law, son-in-law, godfather, godson,
  great-grandfather, fiance, boyfriend, ex-husband
- **Neutral (None):** cousin, parent, sibling, partner, spouse, child, kid,
  grandchild, grandkid, grandparent, godparent, godchild, twin, relative, in-law,
  stepparent, stepchild, stepkid, ex

Spacing/hyphen variants are tolerated: `in-law` ≈ `in law`, `stepmom` ≈
`step mom`, `grandma` ≈ `grand ma` (prefixes step/grand/great/god/half).

### 3b. Optional descriptive modifiers allowed between possessive and kin word
older, elder, younger, little, big, baby, oldest, youngest, eldest, only, middle,
twin, maternal, paternal, biological, bio, adoptive, adopted, foster, late, dear,
beloved, first, second, third, current, former, ex.

### 3c. Name-span pattern
1–3 capitalized tokens, each allowing internal apostrophes/hyphens
(`"Maria"`, `"Maria Rodriguez"`, `"De'Andre"`, `"O'Brien"`, `"Mary-Jane"`);
deliberately excludes `.` so a match can't run across a sentence boundary.

### 3d. The six surface patterns matched (case-insensitive)
1. **`my aunt Maria`** → interviewee → target (possessives: `my`, `our`).
2. **`his brother John`** → nearest preceding person → target (pronouns: `his`,
   `her`, `their`).
3. **`my cousin named/called Trey`** → interviewee (if `my/our`) or nearest
   preceding person.
4. **`Maria, my aunt` / `Denise, who is his sister`** (appositive) → interviewee
   or nearest preceding person.
5. **`Maria's brother John`** (named possessor) → named person → target.
6. **`my mom's sister Denise`** (possessive chain) → interviewee → target, detail
   recorded as `"mom's sister"`, gender taken from the final kin word.

### 3e. Supporting rules
- **Target resolution:** the named span is linked to whichever existing entity
  its character range overlaps.
- **Antecedent resolution** (patterns 2–4 without `my/our`): the most recent
  `PERSON` entity mentioned *before* the match, excluding the target itself.
- **Dedup:** at most one edge per `(source, target)` pair; no self-edges.
- **Gender inference:** the kin word sets the target's gender **only if** not
  already set.

---

## 4. Person attributes (`graph/attributes.py`)

- **Given/surname split:** from the longest name form, drop `KINSHIP_AND_TITLES`
  tokens; first token → `given_name`, last token → `surname` (single token → only
  `given_name`, surname `None`; middle names ignored).
- **Public-figure list (`PUBLIC_FIGURES`) → `replace=False`, subtype
  `PUBLIC_FIGURE`:** obama, barack obama, michelle obama, biden, joe biden, trump,
  donald trump, hillary clinton, bill clinton, george bush, reagan, nixon, jfk,
  kennedy, mlk, martin luther king, malcolm x, mandela, nelson mandela, putin,
  gandhi, beyonce, jay-z, oprah, oprah winfrey, kanye, drake, rihanna, madonna,
  elvis, michael jackson, taylor swift, kim kardashian, lebron, lebron james,
  michael jordan, kobe, kobe bryant, serena williams, tom brady, muhammad ali.
- **Possessive override on public figures:** if any mention is preceded (within
  40 chars) by `my/our/his/her/their` + up to 2 words, it's treated as a **private**
  person who shares the name — kept for replacement and **flagged**.
- **Professional context (`PROFESSIONAL_CONTEXT`) → subtype `PROFESSIONAL`:**
  triggered by `my/our/the` + one of: caseworker, case worker, social worker,
  doctor, dr, nurse, therapist, counselor, counsellor, psychiatrist, psychologist,
  physician, surgeon, pediatrician, dentist, midwife, teacher, professor,
  instructor, tutor, advisor, adviser, mentor, principal, dean, coach, boss,
  manager, supervisor, landlord, lawyer, attorney, pastor, priest, rabbi, imam,
  chaplain, parole officer, probation officer, po, sponsor, babysitter, nanny,
  caregiver — found within a **60-char** window around any mention.
- **Family subtype:** any entity that is a source or target of a `RELATED_TO`
  edge gets subtype `FAMILY` (unless already `PUBLIC_FIGURE`).
- **Defaults:** `replace` defaults to `True`; `gender` defaults to `None` (left
  for gender-neutral surrogates).
- **LLM augmentation (optional, off by default):** 
  - *Open-world classifier* — when a person is NOT on the `PUBLIC_FIGURES` list
    and isn't family/professional, may add a `candidate_public_figure` **flagged
    suggestion**. Never changes `replace` (stays `True`); only surfaces it.
  - *Attribute inferrer* — a bounded, windowed pass (people tagged and judged
    together within each fixed-size window, so it scales to long transcripts),
    reconciled against these rules: it agrees
    with a rule-set value → keep it; the rule left gender unset → adds
    `suggested_gender` (a suggestion, not the trusted `gender`); it conflicts
    with a rule-set gender → **flag** and keep the rule's value. Also adds a
    `suggested_role` descriptor. 

---

## 5. Locations (`graph/location_dates.py` + `data/gazetteer.csv`)

- **Gazetteer schema:** `name, type, parent, aliases` (aliases `|`-separated).
- **Alias resolution:** a location form is matched to a canonical gazetteer name
  via the alias map (e.g. `NOLA` → `New Orleans`).
- **Subtype assignment:** an entity found in the gazetteer gets subtype =
  `type.upper()` (e.g. `CITY`, `NEIGHBORHOOD`, `INSTITUTION`).
- **`LOCATED_IN` edges:** built from `parent` when the parent is **also** a
  detected entity in the transcript.
- **LLM augmentation (optional, off by default):** a location NOT in the
  gazetteer may get `suggested_type` / `suggested_parent` **flagged suggestions**
  from the open-world classifier.

---

## 6. Dates (`graph/location_dates.py`)

Every date entity gets `shiftable` (default `True`).

### 6a. `DATE_ABSOLUTE` / `DATE_OF_BIRTH`
Parsed with `dateutil` (fuzzy) → ISO `resolved_value`. On failure, flagged
`"absolute date failed to parse"`.

### 6b. `DATE_ANCHOR` → fixed public events (`shiftable=False`)
Matched as a **substring** of the mention (longest phrase first). Table:

| phrase(s) | date |
|---|---|
| hurricane katrina, katrina | 2005-08-29 |
| hurricane sandy | 2012-10-29 |
| hurricane harvey | 2017-08-25 |
| deepwater horizon, bp oil spill | 2010-04-20 |
| fukushima | 2011-03-11 |
| obama got elected, obama was elected, obama's election | 2008-11-04 |
| obama inauguration | 2009-01-20 |
| capitol riot, january 6th, jan 6th | 2021-01-06 |
| covid, the pandemic, the lockdown, quarantine started | 2020-03-01 |
| 9/11, september 11th, sept 11th, twin towers | 2001-09-11 |
| boston marathon bombing | 2013-04-15 |
| george floyd | 2020-05-25 |
| y2k, the new millennium | 2000-01-01 |
| the great recession, financial crisis | 2008-09-15 |
| london olympics | 2012-07-27 |

Unrecognized anchor → flagged `"anchor phrase not in ANCHOR_EVENTS table"`.
**LLM augmentation (optional, off by default):** on such a miss, the open-world
classifier may add a `suggested_value` (ISO date) + `suggested_event` **flagged
suggestion**, never overriding the rule's `resolved_value`. 

### 6c. `DATE_RELATIVE` → resolved against `interview_date` (`approximate=True`)
If no `interview_date` in metadata → flagged. Otherwise, in order:
- **`yesterday`** → interview_date − 1 day; **`today`/`tonight`/`this
  morning`/`this afternoon`** → interview_date; **`tomorrow`** → +1 day.
- **`N <unit> ago`** (units: day/week/month/year/decade; unit-days
  1/7/30/365/3650). `N` may be a digit, a spelled number `one`…`ten`, or a vague
  quantity: **`a couple`(=2)**, **`a few`(=3)**, **`several`(=4)**.
- **`last|this|this past|next <week|month|year|decade>`** → ± one unit
  (`next` is future).
- **`last|this|this past <season>`** → mid-season date in the prior occurrence.
  Seasons: spring `03-20`, summer `06-21`, fall/autumn `09-22`, winter `12-21`.
- **`last|this past <weekday>`** → most recent past occurrence of that weekday.
- No pattern matched → flagged `"relative date pattern not recognized"`.

---

## 7. Ages (`graph/location_dates.py`)

Stored as `value`. Tried in order:
- **Numeric:** first 1–3 digit run (`"42 years old"` → 42).
- **Decade expressions** (`approximate=True`): `(early|mid|middle|late)? <decade>`
  where decade ∈ twenties…nineties. Representative age = decade base + bump
  (`early`=+2, `mid`/`middle`=+5, `late`=+8, none=+5). E.g. `"mid-thirties"` → 35.
- **`<tens>-something`** (`approximate=True`): e.g. `"twenty-something"` → 25.
- **Spelled-out sum:** tokens from ones (one…nine), teens (ten…nineteen), tens
  (twenty…ninety) are summed, e.g. `"thirty seven"` → 37.
- None matched → flagged `"could not parse age value"`.

---

## 8. Age ↔ date consistency (`age_date_constraints`)

- **`STATED_WITH` edges:** an `AGE` entity is linked to every
  `DATE_ABSOLUTE`/`DATE_RELATIVE`/`DATE_ANCHOR` mentioned within **`window`
  sentences** (default = same sentence; sentences split on `. ! ?`). Purpose: keep
  age/date arithmetic consistent after date-shifting downstream.

---

## 9. Entity assembly & graph (`graph/pipeline.py`, `graph/models.py`)

- **Person entities:** from stage 1 + stage 2 (merge/coref).
- **Non-person entities:** LOCATION/INSTITUTION, each `DATE_*`, and AGE are each
  grouped **one entity per distinct (lowercased) surface form**.
- **Interviewee:** a synthetic `PERSON` entity `{transcript_id}_e000`, no detected
  span, `attributes={role: interviewee, replace: True}` — the anchor for
  first-person kinship edges.
- **Relation types defined (`Relation` enum):** `RELATED_TO`, `LOCATED_IN`,
  `NEAR`, `WORKS_AT`, `STATED_WITH`. **Currently produced:** `RELATED_TO`
  (kinship), `LOCATED_IN` (gazetteer), `STATED_WITH` (age/date). **Not yet
  produced:** `NEAR`, `WORKS_AT`.

---

## 10. Review flags produced (`needs_review`)

An entity is flagged for human review by any of:
- short name matches multiple long names (ambiguous containment),
- bare given name also appears with a title elsewhere (honorific collision),
- coref linked two entities that co-occur in a sentence — not merged (LLM off),
- coref linked two entities with conflicting genders — not merged,
- coref suggests same but names differ — not merged,
- coref linked two entities but the LLM did not confirm same person — not merged
  (LLM on),
- name matches a public figure but is used possessively — treated as private,
- absolute date failed to parse,
- anchor phrase not in the events table,
- relative date but no `interview_date`,
- relative date pattern not recognized,
- could not parse age value.

Plus, when the optional LLM layer is on:
- LLM suggests a person is a public figure — review whether to keep unredacted,
- LLM-suggested location type/parent — review,
- LLM-suggested anchor date — review,
- LLM-inferred gender conflicts with the rule-derived gender — kept rule value.

---

## 11. Labels ingested but **not** yet graphed

These validate on input but the pipeline does not currently build entities,
edges, or attributes for them: `PHONE`, `EMAIL`, `SSN_OR_ID`, `USERNAME_HANDLE`,
`OCCUPATION`. (`OCCUPATION` context words do inform the `PROFESSIONAL` subtype of
nearby people, but no occupation entity is created.)