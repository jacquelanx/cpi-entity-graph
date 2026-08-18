"""
Speaker-turn segmentation.

Every first-person cue this pipeline reads -- "my", "I", "call me", "reach me at"
-- is evidence about the INTERVIEWEE only when the interviewee is the one
talking. A transcript is a dialogue, so scanning it as flat text silently
attributes the interviewer's words to the subject. interview_002 opens with

    INTERVIEWER: I appreciate you having me out.

which contains two first-person cues that no downstream regex could tell apart
from the subject's own speech. Nothing in the pipeline knew who was speaking, so
`infer_interviewee_gender` regexed the whole file and every ownership cue search
could bind across a turn boundary.

This module splits the transcript into turns, labels each one, and offers two
ways to consume that:

  * `mask_to_subject` -- a SAME-LENGTH copy of the transcript in which every
    character NOT spoken by the subject is replaced by NUL. Existing regexes run
    over it unchanged and every offset still points at the original text, so a
    caller becomes turn-aware without rewriting its patterns. NUL is used rather
    than a space so that `\\s+` inside a multi-token pattern cannot match ACROSS
    a masked-out turn and stitch two subject turns together.
  * `role_at` / `in_subject_turn` / `turn_bounds` -- point queries, for
    span-level decisions and for clipping a sentence to its turn.

Degradation is deliberate. A transcript with no recognizable speaker labels
becomes a single UNKNOWN turn, and UNKNOWN counts as subject text, so an
unlabelled transcript behaves exactly as it did before this module existed.

Role assignment:
  INTERVIEWER  a known questioner label ("INTERVIEWER", "Q", "HOST", ...)
  SUBJECT      a known subject label ("SPEAKER", "A", "NARRATOR", ...) or -- when
               the transcript uses no known subject label -- a NAMED label, since
               a two-party transcript labelled "INTERVIEWER:" / "MARIA:" is
               naming the subject.
  OTHER        a named label in a transcript that ALSO uses a known subject
               label. A third voice is not the subject, and guessing that it is
               would hand its first-person speech to the interviewee.
  UNKNOWN      text before the first label, or a transcript with no labels.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from functools import lru_cache

INTERVIEWER, SUBJECT, OTHER, UNKNOWN = "interviewer", "subject", "other", "unknown"

# Labels are matched only at the start of a line: 1-4 capitalized tokens followed
# by a colon. Deliberately anchored -- an unanchored match would fire on "Note:"
# mid-paragraph and split a turn in half.
_LABEL = re.compile(
    r"(?:(?<=\n)|\A)[ \t]*"
    r"((?:[A-Z][A-Za-z.'’\-]*)(?:[ \t]+[A-Z0-9#][A-Za-z0-9.'’\-]*){0,3})"
    r"[ \t]*:[ \t]*")

_INTERVIEWER_WORDS = {
    "interviewer", "interviewers", "int", "iv", "q", "qu", "question",
    "questioner", "host", "researcher", "moderator", "facilitator",
    "fieldworker", "collector",
}
_SUBJECT_WORDS = {
    "speaker", "subject", "narrator", "respondent", "participant", "interviewee",
    "a", "ans", "answer", "informant", "witness", "storyteller",
}
# a trailing enumerator on a role label: "INTERVIEWER 2", "SPEAKER #1"
_ENUM = re.compile(r"(?:[\s_\-]+|\s*#\s*)(?:\d+|ii?i?)$", re.I)


def _key(label: str) -> str:
    return _ENUM.sub("", label.strip().rstrip(".:").lower()).strip()


@dataclass(frozen=True)
class Turn:
    start: int
    end: int
    label: str
    role: str

    def contains(self, pos: int) -> bool:
        return self.start <= pos < self.end


@lru_cache(maxsize=16)
def parse_turns(transcript: str) -> tuple[Turn, ...]:
    """Turns tiling the transcript. Cached: several stages ask for the same one."""
    hits = list(_LABEL.finditer(transcript))
    if not hits:
        return (Turn(0, len(transcript), "", UNKNOWN),)

    keys = [_key(h.group(1)) for h in hits]
    # A named label is the SUBJECT only when no known subject label is present.
    named_role = OTHER if any(k in _SUBJECT_WORDS for k in keys) else SUBJECT

    turns: list[Turn] = []
    if hits[0].start() > 0:
        turns.append(Turn(0, hits[0].start(), "", UNKNOWN))
    for i, h in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(transcript)
        k = keys[i]
        if k in _INTERVIEWER_WORDS:
            role = INTERVIEWER
        elif k in _SUBJECT_WORDS:
            role = SUBJECT
        else:
            role = named_role
        turns.append(Turn(h.start(), end, h.group(1).strip(), role))
    return tuple(turns)


def has_labels(transcript: str) -> bool:
    turns = parse_turns(transcript)
    return not (len(turns) == 1 and turns[0].role == UNKNOWN)


def turn_at(pos: int, turns=None, transcript: str | None = None) -> Turn | None:
    turns = turns if turns is not None else parse_turns(transcript or "")
    for t in turns:
        if t.contains(pos):
            return t
    return turns[-1] if turns else None


def role_at(pos: int, turns=None, transcript: str | None = None) -> str:
    t = turn_at(pos, turns, transcript)
    return t.role if t is not None else UNKNOWN


def in_subject_turn(pos: int, turns=None, transcript: str | None = None) -> bool:
    """True when `pos` is text the INTERVIEWEE spoke (or the turn structure is
    unknown, in which case we keep the pre-turn-awareness behaviour)."""
    return role_at(pos, turns, transcript) in (SUBJECT, UNKNOWN)


def in_interviewer_turn(pos: int, turns=None, transcript: str | None = None) -> bool:
    return role_at(pos, turns, transcript) == INTERVIEWER


def turn_bounds(pos: int, turns=None, transcript: str | None = None) -> tuple[int, int]:
    t = turn_at(pos, turns, transcript)
    return (t.start, t.end) if t is not None else (0, 0)


MASK_CHAR = "\x00"


@lru_cache(maxsize=16)
def mask_to_subject(transcript: str) -> str:
    """Same-length copy with every non-subject character replaced by NUL.

    Offsets are preserved, so a match found here can be reported against the real
    transcript. The label itself ("SPEAKER:") is masked too -- it is metadata, not
    speech, and leaving it in let a named label be read as a name the subject
    said aloud.
    """
    turns = parse_turns(transcript)
    if len(turns) == 1 and turns[0].role in (SUBJECT, UNKNOWN):
        return transcript
    out = [MASK_CHAR] * len(transcript)
    for t in turns:
        if t.role not in (SUBJECT, UNKNOWN):
            continue
        body = t.start
        if t.label:                       # skip past "LABEL:" into the speech
            colon = transcript.find(":", t.start)
            if 0 <= colon < t.end:
                body = colon + 1
        for i in range(body, t.end):
            out[i] = transcript[i]
    return "".join(out)


def subject_labels(transcript: str) -> list[str]:
    """Distinct NAMED subject labels, most-spoken first.

    A transcript labelled "INTERVIEWER:" / "OPAL:" names its subject in the label
    column; `interviewee.py` uses that as one identification signal.
    """
    spoken: dict[str, int] = {}
    for t in parse_turns(transcript):
        if t.role != SUBJECT or not t.label or _key(t.label) in _SUBJECT_WORDS:
            continue
        spoken[t.label] = spoken.get(t.label, 0) + (t.end - t.start)
    return sorted(spoken, key=lambda k: -spoken[k])
