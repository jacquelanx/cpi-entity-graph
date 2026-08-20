"""
Regression guards for the decisions interviewee-only surrogate generation depends on.

PURPOSE
    Pin the specific behaviours that have broken before. Each test corresponds to a
    bug that actually shipped, so a failure here means that failure is back.

FIT
    Imports the rule parsers, the checkers, `second_line.owner_survivors` and the
    serializer directly, and exercises them on tiny hand-built entities -- no
    transcript files, no fixtures, no network, no LLM. `scripts/eval.py` covers the
    end-to-end numbers; this covers the invariants that made those numbers wrong.

HOW -- a deliberately minimal harness
    No pytest. `check(label, got, want)` compares and records, `FAILS` accumulates
    failures, and `main` runs every test then returns an exit code. Each test builds
    the smallest thing the code under test needs -- usually one `Entity` and a
    `CheckContext` around a one-sentence transcript -- via a small local helper.

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
  temporal       AGE and DATE entities reached surrogate generation with NO `replace`
                 key -- AGE with no redaction directive of any kind -- so a consumer
                 keying off `replace` printed the speaker's date of birth and every
                 age verbatim.
  serialization  `Relation` did not compare equal to its wire value, so
                 `serialize.location_chain` never walked a LOCATED_IN edge.
  artifact       nothing ever WROTE a graph to disk, and the writer that existed
                 dropped the review gate and pinned its offsets to no particular
                 text -- so a re-transcription would have spliced surrogates at the
                 wrong positions with no error anywhere.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.checks import CheckContext
from graph.rules.dates import parse_absolute_date, parse_spoken_year
from graph.rules.ages import parse_age_value
from graph.rules.name_matching import split_name_parts
from graph.models import Edge, Entity, Mention, Relation
from graph.second_line import POLICIES, owner_survivors
from graph.serialize import build_nx_graph, location_chain

FAILS: list[str] = []


def check(label, got, want):
    """Assert `got == want`, printing the result and recording any failure.

    Deliberately does NOT raise: every case runs on every invocation, so one
    regression does not hide the others. `main` reads `FAILS` at the end and turns
    a non-empty list into a non-zero exit code.
    """
    if got != want:
        FAILS.append(f"{label}\n      got  {got!r}\n      want {want!r}")
        print(f"  FAIL  {label}")
    else:
        print(f"  ok    {label}")


# --------------------------------------------------------------------- ownership
def _owner(text, span, label="AGE", others=()):
    """Which owner directions the deterministic checkers support for `span`.

    Builds the smallest world the ownership checkers need: one entity covering
    `span`, one PERSON entity per name in `others` (so "a named person intervenes"
    can be exercised), and a synthetic interviewee. Offsets come from
    `text.index(span)`, which is why the test strings keep each span unique.

    Returns the surviving directions -- `["interviewee"]`, `["other"]`, `[]` for
    ambiguous, or both when the text supports neither exclusively.
    """
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
    """GUARD: the speaker's own identifiers and ages must be attributable to them.

    The recall half checks that a first-person cue several sentences back, in the
    same turn, still binds a span. The precision half checks that the four guards
    -- a kin noun, a named person, a third-person subject, a turn boundary -- still
    refute. Both directions matter: the failure was the speaker's own email and
    life-course ages coming out BLOCKING.
    """
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
    """GUARD: a lone name token behind a form of address is a SURNAME.

    "Father Nguyen" must not yield `given_name="Nguyen"` -- the surrogate generator
    would mint a fake FIRST name to stand in for a family name. Also checks the
    ambiguous kin/title words ("Father", "Sister") abstain rather than guess.
    """
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
    """GUARD: season-plus-year and spoken-year expressions parse correctly.

    "spring of 1975" must resolve to the season, not January 1st (a 78-day error
    that, under RULE_WINS, beat the model's correct answer), and a spelled-out year
    like "nineteen and sixty" must resolve rather than falling through to the LLM.
    """
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
    """GUARD: the kin-word tables stay SYMMETRIC across dialect variants.

    "mamaw" was missing where "papaw" was present, so the speaker's grandmother got
    no relation edge, no FAMILY subtype and no rule gender while his grandfather got
    all three. Checks the same words are known to every table that needs them.
    """
    print("\nkin vocabulary -- mamaw/papaw symmetry")
    from graph.checks.comparators import kin_canon
    from graph.checks.relation_evidence import _GROUP_OF
    from graph.rules.kinship import KINSHIP_GENDER
    from graph.rules.name_matching import KINSHIP_AND_TITLES

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
    """GUARD: a measurement must not be adopted as somebody's AGE.

    "the water came up twelve feet" parses as 12, is in range, and would otherwise
    acquire an owner and become a real age in the graph. The word directly after the
    span is what refutes it -- while "sixty-eight years old" must still pass.
    """
    print("\nmeasurements are not ages")
    from graph.checks.ages import not_a_measurement

    def outcome(text, span):
        """Run `not_a_measurement` over one AGE span and return its `CheckOutcome`."""
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


# --------------------------------------------------- temporal redaction directive
def _temporal_ctx(text, span, category, attrs=None, provenance=None):
    """One AGE/DATE entity plus a `CheckContext` around it, as `(entity, ctx)`.

    `attrs` seeds the entity's attributes and `provenance` its decision records --
    both needed here because the redaction checkers read them (`replace_age`
    consults the `value` Resolution, for instance).
    """
    i = text.index(span)
    e = Entity("t_X1", category,
               mentions=[Mention("t", i, i + len(span), span, category, "t_m1")],
               attributes=dict(attrs or {}))
    e.provenance = dict(provenance or {})
    iv = Entity("t_e000", "PERSON", attributes={"role": "interviewee"})
    ctx = CheckContext(transcript=text, entities=[e], edges=[], interviewee=iv,
                       entity=e)
    return e, ctx


def test_temporal_redaction():
    """GUARD: every AGE and DATE reaches the consumer with a `replace` directive.

    They once arrived with none at all -- AGE with no redaction directive of any
    kind -- so a consumer keying off `replace` printed the speaker's date of birth
    and every age verbatim. Also checks the keep gate: only a span positively
    refuted as an age may survive.
    """
    print("\ntemporal redaction -- dates and ages carry a `replace` directive")
    from graph.checks.ages import (age_reading_refuted,
                                   keep_not_an_explicit_age_phrase,
                                   keep_only_if_refuted_as_an_age)
    from graph.checks.dates import (anchor_phrase_for, keep_is_not_a_local_event,
                                    keep_only_if_public_event)
    from graph.second_line import Resolution, _fields_for, _rule_value

    # the shared table match the rule layer and three checkers now agree on
    check("anchor phrase is article-insensitive",
          anchor_phrase_for("the Great Recession"), "the great recession")
    check("anchor phrase takes the longest match",
          anchor_phrase_for("Hurricane Katrina"), "hurricane katrina")
    check("a private date names no anchor",
          anchor_phrase_for("March 4th, 1951"), "")

    # ---- dates: the rule is the resolved `shiftable`, the keep gate is the table
    e, ctx = _temporal_ctx("It was right after 9/11 that things changed.", "9/11",
                           "DATE_ANCHOR", {"shiftable": False})
    check("a non-shiftable anchor is the rule's keep",
          _rule_value(e, POLICIES["replace_date"], ctx), False)
    check("a national anchor may be kept",
          (keep_only_if_public_event(False, ctx).passed,
           keep_is_not_a_local_event(False, ctx).passed), (True, True))

    e, ctx = _temporal_ctx("We lost the house in the Buffalo Creek flood.",
                           "Buffalo Creek flood", "DATE_ANCHOR", {"shiftable": False})
    check("a REGIONAL anchor may not be kept -- it pins one community",
          keep_is_not_a_local_event(False, ctx).passed, False)

    e, ctx = _temporal_ctx("I was born March 4th, 1951 up the holler.",
                           "March 4th, 1951", "DATE_OF_BIRTH", {"shiftable": True})
    check("a shiftable date is replaced by rule",
          _rule_value(e, POLICIES["replace_date"], ctx), True)
    check("a private date may not be kept",
          keep_only_if_public_event(False, ctx).passed, False)

    # ---- ages: keep ONLY a span a deterministic check proved is not an age
    refuted = {"value": Resolution("value", "conflict", None,
                                   checks_failed=("not_a_measurement",))}
    unusable = {"value": Resolution("value", "reject", None,
                                    checks_failed=("plausible_age_range",))}

    e, ctx = _temporal_ctx("the water came up twelve feet, and we prayed.", "twelve",
                           "AGE", provenance=refuted)
    check("a measurement is not an age, so its text stays",
          _rule_value(e, POLICIES["replace_age"], ctx), False)
    check("...and the keep clears both checkers",
          (keep_only_if_refuted_as_an_age(False, ctx).passed,
           keep_not_an_explicit_age_phrase(False, ctx).passed), (True, True))

    e, ctx = _temporal_ctx("Now I'm sixty-eight and I couldn't leave.", "sixty-eight",
                           "AGE", {"value": 68})
    check("a real age is replaced by rule",
          _rule_value(e, POLICIES["replace_age"], ctx), True)
    check("...and a keep on it is refuted",
          keep_only_if_refuted_as_an_age(False, ctx).passed, False)

    # an age nobody could USE is still an age: an out-of-range value is not a licence
    # to print the span
    e, ctx = _temporal_ctx("She was a hundred and twenty when she passed.",
                           "a hundred and twenty", "AGE", provenance=unusable)
    check("an unusable VALUE does not license keeping the span",
          age_reading_refuted(e), "")
    check("...so the rule still replaces it",
          _rule_value(e, POLICIES["replace_age"], ctx), True)

    # ...and neither does a refutation, if the text plainly reads as an age
    e, ctx = _temporal_ctx("He was sixty-eight years old that winter.", "sixty-eight",
                           "AGE", provenance=refuted)
    check("'years old' refutes a keep whatever the value resolution said",
          keep_not_an_explicit_age_phrase(False, ctx).passed, False)

    # ---- wiring: both fields are resolved, in the right order, onto `replace`
    a, d = Entity("t_A1", "AGE"), Entity("t_D1", "DATE_OF_BIRTH")
    iv = Entity("t_e000", "PERSON")
    fa, fd = _fields_for(a, iv), _fields_for(d, iv)
    check("AGE resolves replace_age", "replace_age" in fa, True)
    check("DATE resolves replace_date", "replace_date" in fd, True)
    check("replace_age writes the `replace` attribute",
          POLICIES["replace_age"].attr, "replace")
    check("replace_date writes the `replace` attribute",
          POLICIES["replace_date"].attr, "replace")
    check("replace_age is ordered after `value`",
          fa.index("replace_age") > fa.index("value"), True)
    check("replace_date is ordered after `shiftable`",
          fd.index("replace_date") > fd.index("shiftable"), True)


# ---------------------------------------------------------------- serialization
def test_serialization():
    """GUARD: `Relation` compares and formats as its wire value.

    Because it mixes in `str`, `Relation.LOCATED_IN == "LOCATED_IN"` must hold --
    otherwise `serialize.location_chain`'s edge filter matches nothing and the
    LOCATED_IN walk surrogate generation needs is silently dead code.
    """
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


# ------------------------------------------------------- titled speaker address
def test_titled_address():
    """GUARD: an honorific must not be treated as a sentence boundary.

    "Thank you for sitting down with me, Ms. Reyes." has to stay ONE sentence, or
    the fragment holding the name carries no second-person cue and the whole
    titled-address route goes dark -- including the only checker that can
    positively confirm the speaker's own gender.
    """
    print("\ntitled address -- an honorific must not split the sentence")
    from graph.checks.gender import interviewee_honorific_address_agrees
    from graph.rules.interviewee import support_for

    text = ("INTERVIEWER: Thank you for sitting down with me, Ms. Reyes. Could we "
            "start with where you grew up?\nSPEAKER: Happy to do it.\n")
    i = text.index("Reyes")
    iv = Entity("t_e000", "PERSON",
                mentions=[Mention("t", i, i + 5, "Reyes", "PERSON", "t_m1")],
                attributes={"role": "interviewee"})

    # `_is_address` used to split the sentence with a naive [.?!]+ scanner, so the
    # period in "Ms." cut the name away from the second-person cue and EVERY titled
    # interviewer address was rejected -- disabling the one checker that can positively
    # confirm the speaker's own gender.
    check("the interviewer's titled address is recognised",
          support_for(iv, text)[0], "address")
    ctx = CheckContext(transcript=text, entities=[], edges=[], interviewee=iv,
                       entity=iv)
    check("...so a gendered honorific confirms the speaker's gender",
          interviewee_honorific_address_agrees("F", ctx).passed, True)
    check("...and refutes the wrong one",
          interviewee_honorific_address_agrees("M", ctx).passed, False)

    # a name the interviewer merely REFERS to is still not an address
    text2 = ("INTERVIEWER: Tell me about Ms. Reyes, the woman next door.\n"
             "SPEAKER: She kept to herself.\n")
    j = text2.index("Reyes")
    other = Entity("t_p9", "PERSON",
                   mentions=[Mention("t", j, j + 5, "Reyes", "PERSON", "t_m9")])
    check("a name being referred to is not an address",
          support_for(other, text2)[0], "")


# -------------------------------------------------------------------- artifact
def test_artifact():
    """GUARD: the output contract -- a bad artifact must be REJECTED, not written.

    Builds a valid payload, then breaks one thing at a time and requires
    `validate_payload` to refuse each: a different source transcript, a mention
    whose offsets no longer slice to its text, an edge or blocking row pointing at
    no entity, a missing interviewee, a version mismatch.
    """
    print("\nartifact -- the contract with surrogate generation")
    import copy
    from graph.loader import Violation
    from graph.serialize import build_payload, validate_payload

    text = "SPEAKER: My name's Rosa and I'm sixty-eight."
    i, j = text.index("Rosa"), text.index("sixty-eight")
    iv = Entity("t_e000", "PERSON",
                mentions=[Mention("t", i, i + 4, "Rosa", "PERSON", "t_m1")],
                attributes={"role": "interviewee", "replace": True})
    age = Entity("t_A1", "AGE",
                 mentions=[Mention("t", j, j + 11, "sixty-eight", "AGE", "t_m2")],
                 attributes={"value": 68, "replace": True})
    info = {"interviewee": iv, "coref_ran": False, "llm_ran": True,
            "llm_model": "test", "interview_date": None,
            "blocking": [("t_e000", "interviewee_gender", "unverified")]}
    payload = build_payload("t", text, [iv, age],
                            [Edge("t_A1", "t_e000", Relation.ATTRIBUTE_OF, "AGE")],
                            info)

    check("a well-formed artifact validates",
          validate_payload(payload, text) is payload, True)
    check("the review gate survives serialization",
          payload["blocking"][0]["field"], "interviewee_gender")
    check("the interviewee is identified explicitly, not by convention",
          payload["interviewee_id"], "t_e000")
    check("offsets are pinned to a specific text",
          len(payload["source"]["sha256"]), 64)

    def rejected(mutate=None, transcript=text) -> bool:
        """True if `validate_payload` REJECTS a (optionally corrupted) copy of the payload.

        `mutate` receives a deep copy of the valid payload and breaks one thing in
        it; the deep copy is what keeps each case independent. Every corruption
        below must be rejected, so the expected answer is always True.
        """
        p = copy.deepcopy(payload)
        if mutate is not None:
            mutate(p)
        try:
            validate_payload(p, transcript)
            return False
        except Violation:
            return True

    check("an artifact built from a DIFFERENT transcript is rejected",
          rejected(transcript=text.replace("Rosa", "Rose")), True)
    check("a mention whose offsets no longer slice to its text is rejected",
          rejected(lambda p: p["entities"][1]["mentions"][0].update(
              {"start": 0, "end": 5})), True)
    check("an edge pointing at no entity is rejected",
          rejected(lambda p: p["edges"][0].update({"target": "t_nope"})), True)
    check("a blocking row nobody can resolve is rejected",
          rejected(lambda p: p["blocking"][0].update({"entity_id": "t_nope"})), True)
    check("an artifact with no interviewee node is rejected",
          rejected(lambda p: p.update({"interviewee_id": None})), True)
    check("a version mismatch is rejected",
          rejected(lambda p: p.update({"graph_version": "0.1"})), True)


def main():
    """Run every test and return a shell exit code: 0 if all invariants hold, 1 if not."""
    for t in (test_ownership, test_name_parts, test_dates, test_kin_vocabulary,
              test_measurements, test_temporal_redaction, test_titled_address,
              test_serialization, test_artifact):
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
