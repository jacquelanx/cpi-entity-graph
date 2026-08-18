"""
Regression guards for the decisions interviewee-only surrogate generation depends on.

    ./venv/bin/python3 tests/test_invariants.py

No pytest, no fixtures, no network, no LLM -- every case here is a DETERMINISTIC
assertion about the rule layer and the checkers, so it runs in under a second and
cannot go green or red because a model changed its mind. `scripts/eval.py` covers the
end-to-end numbers; this covers the invariants that made those numbers wrong.

Each case is a bug that actually shipped. If one of these fails, the corresponding
failure is back:

  ownership      the speaker's own email / life-course ages were BLOCKING because the
                 first-person cue lookback stopped at one sentence -- while the guards
                 (a kin noun, a named person, a third-person subject, a turn boundary)
                 are what carry the precision and must still refute.
  name parts     "Father Nguyen" -> given_name "Nguyen": a lone token behind a form of
                 address is a SURNAME, and the surrogate generator would otherwise mint
                 a fake FIRST name for it.
  dates          "spring of 1975" -> 1975-01-01 (78 days out, and RULE_WINS meant the
                 model's correct answer lost the conflict); "nineteen and sixty" -> None.
  kin vocabulary "mamaw" was missing from the table that builds the KIN regex while
                 "papaw" was present, so the speaker's grandmother got no relation
                 edge, no FAMILY subtype and no rule gender.
  measurements   "the water came up twelve feet" was adopted as somebody's AGE.
  serialization  `Relation` did not compare equal to its wire value, so
                 `serialize.location_chain` never walked a LOCATED_IN edge.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.checks import CheckContext
from graph.location_dates import parse_absolute_date, parse_age_value, parse_spoken_year
from graph.merge_strings import split_name_parts
from graph.models import Edge, Entity, Mention, Relation
from graph.second_line import POLICIES, owner_survivors
from graph.serialize import build_nx_graph, location_chain

FAILS: list[str] = []


def check(label, got, want):
    if got != want:
        FAILS.append(f"{label}\n      got  {got!r}\n      want {want!r}")
        print(f"  FAIL  {label}")
    else:
        print(f"  ok    {label}")


# --------------------------------------------------------------------- ownership
def _owner(text, span, label="AGE", others=()):
    """Which owner directions the deterministic checkers support for `span`."""
    i = text.index(span)
    ents = [Entity("t_A1", label,
                   mentions=[Mention("t", i, i + len(span), span, label, "t_m1")])]
    for n, name in enumerate(others):
        j = text.index(name)
        ents.append(Entity(f"t_p{n}", "PERSON",
                           mentions=[Mention("t", j, j + len(name), name,
                                             "PERSON", f"t_pm{n}")]))
    iv = Entity("t_e000", "PERSON", attributes={"role": "interviewee"})
    ctx = CheckContext(transcript=text, entities=ents, edges=[], interviewee=iv,
                       entity=ents[0])
    return owner_survivors(POLICIES["owner"], ctx)


def test_ownership():
    print("\nownership -- whose span is this?")
    # RECALL: a cue several sentences back, in the same turn, still binds.
    check("speaker's own email, cue two sentences back",
          _owner("SPEAKER: Oh, it's a business, we're easy to find. The shop line is "
                 "228-555-0143. Email's a@b.net.", "a@b.net", "EMAIL"),
          ["interviewee"])
    check("speaker's own age, cue three short sentences back",
          _owner("SPEAKER: I did. Course I did. What else was there. Went in at "
                 "eighteen, came out at fifty-five.", "eighteen"),
          ["interviewee"])
    check("second age in the same sentence",
          _owner("SPEAKER: I did. Course I did. What else was there. Went in at "
                 "eighteen, came out at fifty-five.", "fifty-five"),
          ["interviewee"])

    # PRECISION: the guards must still refute, or the widened reach is a leak.
    check("kin noun between cue and span -> the relative's",
          _owner("SPEAKER: I put our kids up. My daughter Trang was maybe twelve then.",
                 "twelve", others=("Trang",)),
          ["other"])
    check("named person nearer than the cue -> theirs",
          _owner("SPEAKER: I remember it. My father, Earl, went in the mines at "
                 "fourteen.", "fourteen", others=("Earl",)),
          ["other"])
    check("third-person subject governs the span -> theirs",
          _owner("SPEAKER: I knew her well. She was a schoolteacher before she "
                 "married.", "schoolteacher", "OCCUPATION"),
          ["other"])

    # BOUNDS: a cue may never cross a turn boundary or the character budget.
    check("interviewer's first person cannot bind the subject's span",
          _owner("INTERVIEWER: I appreciate you having me out. My phone is 555-0100.\n"
                 "SPEAKER: The number is 304-555-0176.", "304-555-0176", "PHONE"),
          [])
    check("interviewer echoing a value is not evidence either way",
          _owner("SPEAKER: Yes.\nINTERVIEWER: You said your father started at "
                 "fourteen.", "fourteen"),
          [])
    check("a cue beyond the lookback budget does not reach",
          _owner("SPEAKER: I was born here. " + "The weather turned. " * 12
                 + "The number is 555-0143.", "555-0143", "PHONE"),
          [])


# ------------------------------------------------------------------- name parts
def test_name_parts():
    print("\nname parts -- given vs surname")
    for form, want in [
        ("Mr. Landry", (None, "Landry")),        # honorific + one token -> surname
        ("Dr. Combs", (None, "Combs")),
        ("Ms. Boudreaux", (None, "Boudreaux")),
        ("Reverend Estep", (None, "Estep")),
        ("Governor Barbour", (None, "Barbour")),
        ("Aunt Maria", ("Maria", None)),         # kin + one token -> given name
        ("Papaw Clarence", ("Clarence", None)),
        ("Opal", ("Opal", None)),
        ("Maria Rodriguez", ("Maria", "Rodriguez")),
        ("John L. Lewis", ("John", "Lewis")),
        # both a kin term and a form of religious address -> abstain, never guess
        ("Father Nguyen", (None, None)),
        ("Brother Estep", (None, None)),
    ]:
        check(f"split {form!r}", split_name_parts(form), want)


# ------------------------------------------------------------------------ dates
def test_dates():
    print("\ndate parsing")
    for text, want in [
        ("spring of 1975", ("1975-03-20", True)),     # season, not Jan 1
        ("the winter of 1972", ("1972-12-21", True)),
        ("nineteen and sixty", ("1960-01-01", True)),  # spoken year
        ("nineteen sixty-five", ("1965-01-01", True)),
        ("eighteen ninety", ("1890-01-01", True)),
        ("March 4th, 1951", ("1951-03-04", True)),
        ("1921", ("1921-01-01", True)),
    ]:
        check(f"parse {text!r}", parse_absolute_date(text), want)
    # a two-digit year pivots into the future; reported year-less so it is flagged,
    # and `checks/dates.iso_valid` then refutes the value
    iso, has_year = parse_absolute_date("winter of '72")
    check("two-digit year reported as year-less", has_year, False)
    check("spoken year of a non-year phrase", parse_spoken_year("March 4th"), None)


# ---------------------------------------------------------------- kin vocabulary
def test_kin_vocabulary():
    print("\nkin vocabulary -- mamaw/papaw symmetry")
    from graph.checks.comparators import kin_canon
    from graph.checks.relations import _GROUP_OF
    from graph.kinship import KINSHIP_GENDER
    from graph.merge_strings import KINSHIP_AND_TITLES

    for w in ("mamaw", "mawmaw", "memaw", "meemaw"):
        check(f"{w!r} implies F in KINSHIP_GENDER", KINSHIP_GENDER.get(w), "F")
    for w in ("papaw", "pawpaw", "pappy"):
        check(f"{w!r} implies M in KINSHIP_GENDER", KINSHIP_GENDER.get(w), "M")
    for w in ("mamaw", "papaw", "meemaw", "pawpaw"):
        check(f"{w!r} stripped by normalize()", w in KINSHIP_AND_TITLES, True)
    check("kin_canon('mamaw')", kin_canon("mamaw"), "grandmother")
    check("relation verifier groups mamaw with grandmother",
          _GROUP_OF.get("mamaw"), _GROUP_OF.get("grandmother"))


# ------------------------------------------------------------------ measurements
def test_measurements():
    print("\nmeasurements are not ages")
    from graph.checks.ages import not_a_measurement

    def outcome(text, span):
        i = text.index(span)
        e = Entity("t_A1", "AGE",
                   mentions=[Mention("t", i, i + len(span), span, "AGE", "t_m1")])
        iv = Entity("t_e000", "PERSON")
        ctx = CheckContext(transcript=text, entities=[e], edges=[], interviewee=iv,
                           entity=e)
        return not_a_measurement(12, ctx)

    check("'twelve feet' refuted as an age",
          outcome("the water came up twelve feet, and we prayed.", "twelve").passed,
          False)
    check("'sixty-eight years old' accepted",
          outcome("Now I'm sixty-eight years old and I couldn't leave.",
                  "sixty-eight").passed,
          True)
    check("a bare age accepted",
          outcome("My daughter was maybe twelve then.", "twelve").passed, True)
    check("age parser still reads a decade as approximate",
          parse_age_value("her eighties"), (85, True))


# ---------------------------------------------------------------- serialization
def test_serialization():
    print("\nserialization -- the artifact surrogate generation consumes")
    check("Relation compares equal to its wire value",
          Relation.LOCATED_IN == "LOCATED_IN", True)
    check("Relation formats as its value", f"{Relation.LOCATED_IN}", "LOCATED_IN")
    check("Edge.to_dict emits a string",
          Edge("a", "b", Relation.LOCATED_IN).to_dict()["relation"], "LOCATED_IN")
    g = build_nx_graph(
        [Entity("a", "LOCATION"), Entity("b", "LOCATION"), Entity("c", "LOCATION")],
        [Edge("a", "b", Relation.LOCATED_IN), Edge("b", "c", Relation.LOCATED_IN)])
    check("location_chain walks the hierarchy", location_chain(g, "a"),
          ["a", "b", "c"])


def main():
    for t in (test_ownership, test_name_parts, test_dates, test_kin_vocabulary,
              test_measurements, test_serialization):
        t()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S):\n")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
