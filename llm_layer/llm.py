"""
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
    h = hashlib.sha256(f"{model}\x00{system}\x00{prompt}".encode("utf-8"))
    return h.hexdigest()


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL, timeout: int = 60,
                 cache_path: Path | None = None):
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
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except Exception:
            self._cache = {}            # corrupt/old file -> start fresh

    def save(self) -> None:
        """Atomically persist the successful (non-None) cache entries."""
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
        """True if the Ollama server answers. Probed once, then cached."""
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
        """Return the model's JSON reply as a dict, or None on any failure.
        Successful results are cached (in memory + persisted to disk); failures
        are cached only for this run so they are retried on a later run."""
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
    """Process-wide client so the availability probe happens once."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = LLMClient()
    return _DEFAULT_CLIENT


# Shared helper: short context snippets around an entity's mentions. Used by the
# task modules (merge_adjudicate, openworld). 
def _windows(transcript: str, entity, radius: int = 160, max_snips: int = 3):
    out = []
    for m in entity.mentions[:max_snips]:
        s = max(0, m.start - radius)
        e = min(len(transcript), m.end + radius)
        out.append("..." + transcript[s:e].strip() + "...")
    return out
