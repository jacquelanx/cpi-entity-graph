"""
The Ollama client, its persistent cache, and the shared context-window helper.

PURPOSE
    One place that talks to the model. Handles availability probing, JSON-mode
    requests, retries, and a disk cache so re-running a transcript costs nothing.
    Also provides `_windows`, the helper every task module uses to assemble the
    transcript excerpts a prompt should contain.

FIT
    The bottom of `llm_layer/`: the four task modules (`extract.py`,
    `openworld.py`, `identifier_judge.py`, `merge_adjudicate.py`,
    `interviewee.py`) all go through `LLMClient.judge`. `graph/pipeline.py` builds
    the client via `default_client()` or is handed one.

HOW
    Determinism first: temperature 0, a fixed seed, and `format: "json"` so the
    reply is parseable. Results are then cached under a HASH of
    (model, system, prompt), which is what makes the cache safe to write to disk
    -- see the privacy note below.

Local-LLM layer (optional, OFF by default). Uses local Ollama server.
If Ollama isn't running or the model can't be reached, the pipeline
behaves EXACTLY like the rules-only version. LLM is not required!
Use by passing an LLMClient into run_pipeline, or by setting KG_USE_LLM=1.

Runtime : Ollama (https://ollama.com) `ollama serve` + `ollama pull <model>`
Model   : qwen2.5:7b-instruct  (override with KG_LLM_MODEL)
Server  : http://localhost:11434  (override with OLLAMA_URL)

Calls run at temperature 0 with JSON-constrained output for reproducibility, and
results are cached per (model, system, prompt) within a client instance.
"""


from __future__ import annotations
import atexit
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_MODEL = os.environ.get("KG_LLM_MODEL", "qwen2.5:7b-instruct")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


# Persistent decision cache so re-processing a transcript doesn't re-query the
# model. PRIVACY: entries are keyed by a HASH of (model, system, prompt); the
# raw prompt (which contains transcript text) is NEVER written to disk. Only the
# model's small JSON responses are stored. Disable with KG_LLM_NO_CACHE=1.
_CACHE_PATH = Path(os.environ.get(
    "KG_LLM_CACHE", str(Path(__file__).resolve().parent.parent / ".llm_cache.json")))
_CACHE_ENABLED = os.environ.get("KG_LLM_NO_CACHE") != "1"
_FLUSH_EVERY = 25   # write to disk after this many new entries (also on exit)


def _key(model: str, system: str, prompt: str) -> str:
    """The cache key: a SHA-256 hash of (model, system, prompt).

    Hashing rather than storing the prompt is the privacy mechanism. Prompts
    contain verbatim transcript text; the hash does not, so the on-disk cache
    holds only opaque keys and the model's short JSON answers. The NUL byte
    (`\x00`) separates the three parts so no two different triples can produce
    the same concatenation.
    """
    h = hashlib.sha256(f"{model}\x00{system}\x00{prompt}".encode("utf-8"))
    return h.hexdigest()


class LLMClient:
    """A cached, fail-soft client for a local Ollama server.

    Two behaviours define it. FAIL-SOFT: every method returns None or False
    rather than raising, so an unreachable server makes the pipeline rules-only
    instead of broken. CACHED: successful answers are kept in memory and flushed
    to disk, so a re-run of the same transcript issues no requests at all.
    """

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL, timeout: int = 60,
                 cache_path: Path | None = None):
        """Set up the client and load any existing cache from disk.

        `_available` starts as None meaning "not probed yet" -- distinct from
        False, which means "probed and the server did not answer". The `atexit`
        hook flushes entries accumulated since the last periodic save, so a run
        that ends between flushes does not lose them.
        """
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._available: bool | None = None
        self._cache: dict = {}          # hash -> response dict (None kept only in-memory)
        self._dirty = 0
        self._cache_path = cache_path or (_CACHE_PATH if _CACHE_ENABLED else None)
        self._load()
        if self._cache_path is not None:
            atexit.register(self.save)  # flush any unsaved entries on exit

    def _load(self) -> None:
        """Read the cache file, silently starting fresh if it is missing or corrupt.

        A cache is an optimization, so an unreadable one is not an error -- the
        run just costs more model calls.
        """
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except Exception:
            self._cache = {}            # corrupt/old file -> start fresh

    def save(self) -> None:
        """Atomically persist the successful (non-None) cache entries.

        FAILURES ARE NOT PERSISTED. A None entry means the model errored or
        returned unparseable output, and that is usually transient -- keeping it
        only in memory means the next run retries instead of caching a failure
        forever.

        "Atomically" means: write to a temporary file in the same directory, then
        `os.replace` it over the real one. `os.replace` is atomic on a single
        filesystem, so a crash mid-write leaves the OLD cache intact rather than a
        half-written file. The whole thing is wrapped in a bare `except` because a
        cache write must never take down a pipeline run.
        """
        if self._cache_path is None or not self._dirty:
            return
        keep = {k: v for k, v in self._cache.items() if v is not None}
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self._cache_path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(keep, f)
            os.replace(tmp, self._cache_path)   # atomic
            self._dirty = 0
        except Exception:
            pass                        # cache is best-effort; never crash the run

    def available(self) -> bool:
        """True if the Ollama server answers. Probed once, then cached.

        Hits the cheap `/api/tags` endpoint with a short 3-second timeout, so a
        machine with no Ollama installed pays the cost once per process rather
        than once per call.
        """
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(self.url + "/api/tags")
            with urllib.request.urlopen(req, timeout=3) as r:
                json.loads(r.read().decode())
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def judge(self, prompt: str, system: str | None = None) -> dict | None:
        """Ask the model one question and return its JSON reply as a dict, or None.

        The single entry point every task module uses. Steps:

          1. CACHE. A hit returns immediately -- including a cached None, so a
             failure is not retried within the same run.
          2. AVAILABILITY. No server means None, with no request attempted.
          3. REQUEST. `temperature: 0` and `seed: 0` make the answer
             reproducible; `format: "json"` constrains the model to emit parseable
             JSON; `num_ctx: 4096` caps the context window (which is why
             `_windows` enforces a character budget).
          4. RETRY ONCE. The loop runs at most twice, because the common failure
             is a malformed or empty parse rather than a broken server, and one
             retry usually clears it. Anything that is not a dict is treated as a
             failure.
          5. CACHE THE RESULT and periodically flush to disk.

        Successful results are cached (in memory + persisted to disk); failures
        are cached only for this run so they are retried on a later run.
        """
        key = _key(self.model, system or "", prompt)
        if key in self._cache:
            return self._cache[key]
        if not self.available():
            return None
        payload = {
            "model": self.model, "prompt": prompt, "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 0, "num_ctx": 4096},
        }
        if system:
            payload["system"] = system
        result = None
        for _ in range(2):  # one retry on a bad/empty parse
            try:
                req = urllib.request.Request(
                    self.url + "/api/generate",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode())
                out = json.loads((resp.get("response") or "").strip())
                if isinstance(out, dict):
                    result = out
                    break
            except Exception:
                continue
        self._cache[key] = result
        if result is not None:          # persist only successful decisions
            self._dirty += 1
            if self._dirty >= _FLUSH_EVERY:
                self.save()
        return result


_DEFAULT_CLIENT: LLMClient | None = None


def default_client() -> LLMClient:
    """The process-wide client, created on first use.

    A module-level singleton so the availability probe and the cache load happen
    once per process rather than once per transcript.
    """
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = LLMClient()
    return _DEFAULT_CLIENT


# Shared helper: context snippets around an entity's mentions. Used by the task
# modules (merge_adjudicate, openworld). `radius`/`max_snips` are per-CALL so each
# caller can size context to how non-local its judgment is (a public-figure call
# needs far more than an age parse -- see the call sites). Two refinements over a
# raw character slice:
#   * snap each snippet OUTWARD to sentence boundaries, so the model reads whole
#     sentences instead of fragments cut mid-word;
#   * merge overlapping/adjacent snippets, so several nearby mentions don't waste
#     tokens re-sending the same passage.
# A soft char budget bounds the joined length so a wide radius can never silently
# blow past the model's context window (num_ctx in judge()).
_CTX_CHAR_BUDGET = 6000


def _windows(transcript: str, entity, radius: int = 160, max_snips: int = 3):
    """Transcript excerpts around an entity's mentions, ready to paste into a prompt.

    Returns a list of strings, each a passage of the transcript with "..." marking
    where it was cut. `radius` is how many characters of context to take either
    side of a mention; `max_snips` caps how many mentions contribute.

    HOW, in four steps:
      1. For each of the first `max_snips` mentions, take a `radius`-character
         window either side.
      2. SNAP OUTWARD to sentence boundaries, so the model reads whole sentences
         rather than fragments cut mid-word.
      3. MERGE overlapping or touching windows, so two nearby mentions do not
         send the same passage twice. Sorting first is what makes a single
         backward-looking comparison (`a <= merged[-1][1]`) sufficient.
      4. TRUNCATE to a soft total character budget, so a generous `radius` can
         never silently overflow the model's context window.
    """
    from .extract import _sentences               # local: keep client.py import-light
    sents = _sentences(transcript) or [(0, len(transcript))]
    n = len(transcript)

    def sent_start(pos):
        """The start offset of the sentence containing `pos` (0 if none)."""
        for (ss, se) in sents:
            if ss <= pos < se:
                return ss
        return 0

    def sent_end(pos):
        """The end offset of the sentence containing `pos` (end of text if none)."""
        for (ss, se) in sents:
            if ss <= pos < se:
                return se
        return n

    raw = []
    for m in entity.mentions[:max_snips]:
        a = max(0, m.start - radius)
        b = min(n, m.end + radius)
        raw.append((sent_start(a), sent_end(max(a, b - 1))))

    raw.sort()
    merged = []
    for (a, b) in raw:
        if merged and a <= merged[-1][1]:            # overlaps/touches previous
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    out, used = [], 0
    for (a, b) in merged:
        seg = transcript[a:b].strip()
        if not seg:
            continue
        seg = seg[:max(0, _CTX_CHAR_BUDGET - used)]  # soft budget: never overflow ctx
        if not seg:
            break
        used += len(seg)
        out.append(("" if a == 0 else "...") + seg + ("" if b >= n else "..."))
    return out
