# The LLM layer

An **optional, local, conservative** LLM layer that sits *on top of* the
deterministic ruleset.

> TL;DR: the LLM is a **second opinion, never the only opinion**. It is off by
> default, runs entirely locally (no API calls), and if it isn't running the
> pipeline behaves exactly like the rules-only version.

## The dual decision / security model

Everything the LLM touches is governed by **which class the operation falls in**.

### Class A — destructive / confounding operations → strict AND-gate
Merging two entities, overriding an existing attribute, or removing redaction.
These can *lose or confound* information, so they require **both halves to agree**:

> a deterministic rule must **propose** the action **and** the LLM must
> **confirm** it (with high confidence or a quoted piece of evidence).
> If either is unsure → **do not act; flag for review.**

### Class B — additive / net-new operations → flagged suggestions
Facts the rules *cannot* produce (a location the gazetteer misses, a public
figure not on the list, an inferred attribute). Here "both agree" is impossible
by definition, so instead:

> the LLM may **propose**, but the result is written as a **flagged suggestion**
> (`source: llm`, `needs_review: true`) — surfaced for a human, **never silently
> trusted**, and **never destructive**.

### IMPORTANT SAFETY NOTE 
**The LLM can never cause under-redaction.** It may *add* redaction or *flag*,
but it can never flip `replace: True → False` on its own. Marking someone
"don't replace" (a public figure) still requires the deterministic list or a
human. The failure mode stays one-directional: at worst we over-redact (safe),
never leak.

---

## Where we are now

| LLM use | Status | Where |
|---|---|---|
| **Merge adjudication** (Class A) | **Implemented & validated** | `llm_layer/merge_adjudicate.py` (called by `graph/coref.py`) |
| **Open-world classification** (Class B) | **Implemented & validated** | `llm_layer/openworld.py` |
| **Attribute inference** (Class B) | **Implemented & validated** | `llm_layer/attr_infer.py` |

### What the merge adjudicator does (implemented)
Inside the coref stage, for each pair the coref model links:
1. **Gender conflict** → always a hard block (no merge).
2. **Same sentence** → a hard block *only when the LLM is off*. When the LLM is
   on, the sentence is handed to the model (it distinguishes "we called Roberto
   Beto" from "Sarah's brother Danny").
3. **Merge** only if the LLM confirms same-person with high confidence or an
   evidence quote (**Class A double-gate**). The quote is stored on the surviving
   entity as `merge_evidence`.
4. Otherwise → **flag, do not merge**.

With the LLM **off**, step 2 stays a hard block and step 3 falls back to the
name-compatibility rule.

### What the open-world classifier does (implemented)
Lives in `llm_layer/openworld.py`, run as a pipeline pass **after** the deterministic
stages. It is pure **Class B**: it fires only on a **list MISS**, only acts on
**high-confidence** answers, and writes **flagged suggestions**.

1. **Person → public figure?** For a person not on `PUBLIC_FIGURES` and not
   family/professional, it may add `candidate_public_figure` + a review flag.
   `replace` stays `True`; a human decides whether to keep the name.
2. **Location → type / parent?** For a place not in the gazetteer, it may add
   `suggested_type` and `suggested_parent` + a review flag (not a trusted subtype
   or `LOCATED_IN` edge).
3. **Anchor date → fixed date?** For a `DATE_ANCHOR` the events table missed, it
   may add `suggested_value` (ISO) + `suggested_event` + a review flag, without
   overriding any rule-resolved value.

### What the attribute inferrer does (implemented)
Lives in `llm_layer/attr_infer.py`. Rather than interrogating each person in
isolation (which makes a small model confuse neighbours), it judges people
**together in context** — but in a **bounded, chunked** way so it scales to long
transcripts:

- walk the transcript in fixed-size **windows** (`_WINDOW_CHARS`, sentence-aligned);
- for each window, run ONE roster call over just the people present in it, with
  **every mention tagged by a local id** (`[P2 Ronnie]`) so the model can tell
  them apart *within* the window;
- accumulate per-entity **votes** across windows (windows never overwrite each
  other) and reconcile once at the end by majority vote;
- an entity is **dropped from later windows** once it has both attributes, or
  after a capped number of queries (`_MAX_VISITS`) — whichever comes first — so a
  person who never fully resolves isn't re-queried in every window they appear in.
  A window whose remaining people are all resolved makes no call at all.

Each call is bounded regardless of transcript length (no single giant/truncated
request); the number of calls is linear in the transcript and cacheable. Short
transcripts fit in one window, so behavior there is unchanged. Reconciliation:

- rule already set gender and the LLM **agrees** → keep it (trusted).
- rule left gender unset → add `suggested_gender` (a **suggestion**, not the
  trusted `gender`).
- LLM gender **conflicts** with a rule-set gender → **flag** and keep the rule's
  value (never silently overridden).
- role → `suggested_role` descriptor (net-new suggestion).

**Scaling.** Windows are size-bounded (`_WINDOW_CHARS`), so no call grows with
transcript length: a 20k-word transcript becomes ~30 small windows (fewer with
early-exit), not one oversized/truncated request. The merge adjudicator and
open-world classifier already scale the same way — small fixed windows, call
count bounded by entity/pair count. The cost is *many small local calls*: with
the LLM on, a long transcript takes on the order of minutes locally. It's linear,
and decisions are **cached to disk across runs** (see the persistent cache under
Setup — a warm re-run was ~30× faster). To trade quality for speed on a cold run,
use a smaller model (e.g. `qwen2.5:3b-instruct`) or run the LLM only on
flagged/uncertain entities.

---

## Setup & usage

**Runtime:** [Ollama](https://ollama.com) (local, offline, Metal-accelerated on
Mac). No extra Python dependencies — the client uses only the stdlib.

```bash
ollama serve                          # start the local server
ollama pull qwen2.5:7b-instruct       # the default model
```

**Enable it** (off by default) via env var — works for the eval and the dashboard:

```bash
KG_USE_LLM=1 ./venv/bin/python3 scripts/eval.py
KG_USE_LLM=1 ./venv/bin/python3 scripts/dashboard.py
```

…or pass a client explicitly:

```python
from graph.llm import default_client
run_pipeline(tid, text, mentions, llm=default_client())
```

**Config (env vars):**
- `KG_USE_LLM=1` — turn the layer on.
- `KG_LLM_MODEL` — model tag (default `qwen2.5:7b-instruct`).
- `OLLAMA_URL` — server URL (default `http://localhost:11434`).
- `KG_LLM_CACHE` — path to the persistent cache (default `.llm_cache.json` at
  the repo root).
- `KG_LLM_NO_CACHE=1` — disable persistence (in-memory only for the run).

**Model choice.** `qwen2.5:7b-instruct` is the default. Llama-3.1-8B
is a fine alternative; a 14B model gives better judgment if you have the RAM.

**Reproducibility.** Calls run at `temperature 0`, `seed 0`, with JSON-constrained
output. Log the model tag alongside the graph version when you record results — a
different local model (or version) can change judgments.

**Persistent cache.** Successful decisions are cached to disk (default
`.llm_cache.json`) keyed by a **hash of `(model, system, prompt)`**, so
re-processing the same transcript re-uses them instead of re-querying the model
(measured ~30× faster on a warm run: 41s → 1.3s per transcript). Only successful
responses are stored; transient failures are retried on the next run. 

**Privacy:** the raw prompt is never written to the cache, only the hash
key and the model's JSON response (which can include a short evidence quote),
so the file is transcript-derived data and is **gitignored**. Changing the model
tag naturally invalidates old entries (the key includes it). Delete the file to
reset, or set `KG_LLM_NO_CACHE=1` to skip persistence.