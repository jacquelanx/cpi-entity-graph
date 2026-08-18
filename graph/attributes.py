"""
Fill each entity's attribute dict for the surrogate generator later.
We accept None values if the value is unknown.
"""


from __future__ import annotations
import re
from .models import Edge, Entity, Relation
from .merge_strings import split_name_parts
from .turns import mask_to_subject


# Names that should usually NOT be replaced (public figures give no PII about
# the interviewee). Prefer full names / distinctive surnames over common names to
# avoid colliding with a private individual who shares the name.
PUBLIC_FIGURES = {
    # politicians / historical
    "obama", "barack obama", "michelle obama", "biden", "joe biden",
    "trump", "donald trump", "hillary clinton", "bill clinton",
    "george bush", "reagan", "nixon", "jfk", "kennedy",
    "mlk", "martin luther king", "malcolm x", "mandela", "nelson mandela",
    "putin", "gandhi",
    # entertainers
    "beyonce", "jay-z", "oprah", "oprah winfrey", "kanye", "drake",
    "rihanna", "madonna", "elvis", "michael jackson", "taylor swift",
    "kim kardashian",
    # athletes
    "lebron", "lebron james", "michael jordan", "kobe", "kobe bryant",
    "serena williams", "tom brady", "muhammad ali",
}


# Signals that a name matching PUBLIC_FIGURES is actually a PRIVATE individual who
# merely shares the name (so it must still be redacted): a first-person possessive
# directly on the name ("my Beyonce"), a naming construction that introduces a
# person ("her name is Beyonce", "a friend named Beyonce"), or an explicit
# "no relation to the famous one" aside. Deliberately NARROW: third-person
# ATTRIBUTIVE use of a famous name ("his Elvis impression", "my dad admired Obama")
# is NOT a personal signal -- the old broad window over-redacted those. Kinship and
# professional context are stronger signals and checked separately.
_POSS_NAME = re.compile(r"\b(?:my|our)\s+$", re.I)
_NAMING = re.compile(r"\b(?:name\s+(?:is|was)|named)\s+$", re.I)
_NOT_FAMOUS = re.compile(r"^\W{0,3}(?:no relation|not the (?:famous|real|actual|celebrity))", re.I)


PROFESSIONAL_CONTEXT = re.compile(
    r"\b(my|our|the)\s+(caseworker|case worker|social worker|doctor|dr\.?|"
    r"nurse|therapist|counselor|counsellor|psychiatrist|psychologist|"
    r"physician|surgeon|pediatrician|dentist|midwife|teacher|professor|"
    r"instructor|tutor|advisor|adviser|mentor|principal|dean|coach|boss|"
    r"manager|supervisor|landlord|lawyer|attorney|pastor|priest|rabbi|imam|"
    r"chaplain|parole officer|probation officer|po|sponsor|babysitter|"
    r"nanny|caregiver)\b", re.I)


# ------------------------- interviewee gender -------------------------
# First-person self-description of the SPEAKER'S OWN gender. We deliberately use
# ONLY gendered nouns the interviewee applies to THEMSELVES ("I'm a mother",
# "call me Grandpa") -- never relational terms about other people ("my husband"),
# which would encode an assumption about the speaker's own gender/orientation. The
# trigger is anchored to an explicit first-person "I ..." (or a "call me <kin>"),
# so third-person predicates ("he was known as a good man") never leak in.
_IV_TRIG = (r"I['’]?m|I am|I was|I['’]ve been|I have been|I became|I['’]d become|"
            r"I grew up|I ended up|I['’]ll be")
# article / possessive introducing the predicate noun ("I'm a mother", "I'm her mom")
_IV_SLOT = r"(?:a|an|the|her|his|their|your|our|my)\s+"
# optional adjectives between the article and the noun ("I was the only girl")
_IV_MOD = (r"(?:(?:only|oldest|youngest|eldest|middle|single|young|old|little|proud|"
           r"new|first|second|working|widowed|divorced|stay[\s-]at[\s-]home|good|"
           r"poor|country|former)\s+){0,3}")
_IV_NOUN_F = (r"woman|girl|lady|gal|mother|mom|mommy|mum|mama|momma|grandmother|"
              r"grandma|granny|nana|grandmom|gramma|daughter|sister|widow|housewife")
_IV_NOUN_M = (r"man|boy|guy|gentleman|fella|fellow|father|dad|daddy|papa|grandfather|"
              r"grandpa|granddad|gramps|son|brother|widower")
_IV_SELF_F = re.compile(r"\b(?:" + _IV_TRIG + r")\s+" + _IV_SLOT + _IV_MOD +
                        r"(?:" + _IV_NOUN_F + r")\b", re.I)
_IV_SELF_M = re.compile(r"\b(?:" + _IV_TRIG + r")\s+" + _IV_SLOT + _IV_MOD +
                        r"(?:" + _IV_NOUN_M + r")\b", re.I)
# "(the kids) call me Grandma" / "everybody calls me Pop"
_IV_CALLME_F = re.compile(r"\bcall(?:ed|s)?\s+me\s+(?:their\s+)?(?:mom|mommy|mama|"
                          r"momma|mother|grandma|granny|nana|meemaw|gramma)\b", re.I)
_IV_CALLME_M = re.compile(r"\bcall(?:ed|s)?\s+me\s+(?:their\s+)?(?:dad|daddy|papa|"
                          r"pop|pops|papaw|pawpaw|grandpa|gramps|granddad)\b", re.I)


def infer_interviewee_gender(transcript: str, interviewee: Entity) -> None:
    """Set the interviewee's OWN gender (rule layer) from first-person self-
    description. Conflicting cues (both F and M matched) leave it unset and raise a
    review flag -- the LLM second line (extract_pass) then fills or confirms it.
    Never overwrites a gender already set.

    Scanned over the SUBJECT'S TURNS ONLY. This used to regex the whole
    transcript, so an interviewer saying "I'm a mother myself" would have set the
    interviewee's gender -- and interview_002's interviewer opens with "I
    appreciate you having me out", which is first-person speech that is not the
    subject's. `graph.turns.mask_to_subject` preserves offsets, so the evidence
    quote still points at real transcript text.
    """
    if interviewee.attributes.get("gender"):
        return
    spoken = mask_to_subject(transcript)
    genders, evidence = set(), None
    for rx, g in ((_IV_SELF_F, "F"), (_IV_CALLME_F, "F"),
                  (_IV_SELF_M, "M"), (_IV_CALLME_M, "M")):
        m = rx.search(spoken)
        if m:
            genders.add(g)
            if evidence is None:
                evidence = m.group(0).strip()
    if not genders:
        # Second rule source: the INTERVIEWER's honorific form of address ("Thank
        # you, Ms. Boudreaux"). Deterministic, and it assumes nothing about the
        # speaker's relationships -- unlike a spouse term, which this module
        # deliberately refuses to read. Only available once the identification stage
        # has given the speaker a name, so it is inert on a transcript that never
        # names them (both samples). Tried only after self-description, which is the
        # stronger signal.
        g, ev = _honorific_address_gender(transcript, interviewee)
        if g:
            genders, evidence = {g}, ev

    if len(genders) == 1:
        interviewee.attributes["gender"] = next(iter(genders))
        interviewee.attributes.setdefault("gender_evidence", evidence)
    elif len(genders) == 2:
        interviewee.flag_entity("conflicting first-person gender cues for the "
                                "interviewee; left unset for the LLM / a human to resolve")


def _honorific_address_gender(transcript: str, interviewee: Entity):
    """(gender, evidence) implied by a gendered honorific the INTERVIEWER uses to
    address the subject, or (None, None). Shares `interviewee._is_address` and
    `checks/gender.HONORIFIC_GENDER` with the checker that verifies this field, so
    the proposer and the verifier read the same evidence."""
    from .interviewee import ADDRESS_TITLED, _is_address
    from .turns import parse_turns, in_interviewer_turn
    from .checks.gender import honorific_gender
    from .merge_strings import normalize

    toks = set()
    for form in interviewee.sorted_mentions:
        toks |= set(normalize(form))
    if not toks:
        return None, None

    turns = parse_turns(transcript)
    found, ev = set(), None
    for m in ADDRESS_TITLED.finditer(transcript):
        if not in_interviewer_turn(m.start(), turns):
            continue
        if not (set(normalize(m.group(1))) & toks):
            continue
        if not _is_address(transcript, m.start(1), m.end(1), m.group(1),
                           titled=True, phrase_start=m.start()):
            continue
        g = honorific_gender(m.group(0))
        if g:
            found.add(g)
            ev = ev or m.group(0).strip()
    return (next(iter(found)), ev) if len(found) == 1 else (None, None)


# ------------------------- role (rule layer) -------------------------
# `role` used to be LLM-only: "nothing to run first, nothing to double-check".
# That was true only because nobody looked -- the rules already compute both
# halves of the answer. A person tied to the interviewee by a kinship edge HAS a
# role (the edge detail), and a person introduced by a professional construction
# HAS one (the matched job word). This function reads those two sources, so `role`
# joins the unified second line like every other field.

def infer_person_role(transcript: str, person_entities: list[Entity],
                      edges: list[Edge]) -> None:
    """RULE layer for `role`: the kinship-edge detail, else a professional cue."""
    detail_for: dict[str, str] = {}
    for ed in edges:
        if ed.relation != Relation.RELATED_TO:
            continue
        # the detail describes the TARGET's role relative to the source
        detail_for.setdefault(ed.target, ed.detail)

    for ent in person_entities:
        if ent.attributes.get("role"):
            continue
        d = detail_for.get(ent.entity_id)
        if d:
            ent.attributes["role"] = d.strip().lower()
            continue
        for m in ent.mentions:
            hit = PROFESSIONAL_CONTEXT.search(
                transcript[max(0, m.start - 60):m.end + 60])
            if hit:
                ent.attributes["role"] = hit.group(2).strip().lower().rstrip(".")
                break


# ---------------------- ethnicity (rule layer) ----------------------
# Also previously LLM-only and, worse, previously UNCHECKED: on the sample
# transcripts every third party inherited the speaker's ethnicity as an
# `inferred` guess from their name alone -- Bao, Hoa, Minh, Thao, Hai, Trang,
# Khanh and Duc were all labelled "Vietnamese (inferred)" on no evidence tied to
# any of them, and Mr. Landry was labelled "Cajun".
#
# Self-identification is a CLOSED construction set, so a rule can own it at high
# precision, exactly like `interviewee.SELF_INTRO` owns self-introduction. The
# label must also be a recognized ethnonym, which is what stops the capture group
# from swallowing an arbitrary capitalized word.
ETHNONYMS = {
    # US regional / heritage identities common in oral history
    "appalachian", "cajun", "creole", "acadian", "gullah", "geechee",
    "pennsylvania dutch", "tejano", "chicano", "hispanic", "latino", "latina",
    "latinx", "melungeon", "scotch-irish", "scots-irish", "ulster scots",
    # broad census-style
    "african american", "afro-caribbean", "black", "white", "asian american",
    "native american", "american indian", "alaska native", "pacific islander",
    "native hawaiian", "mixed race", "biracial", "multiracial", "jewish",
    "arab", "arab american", "roma", "romani",
    # nationality / people terms
    "vietnamese", "cambodian", "laotian", "hmong", "thai", "filipino", "filipina",
    "chinese", "japanese", "korean", "taiwanese", "indian", "pakistani",
    "bangladeshi", "sri lankan", "nepali", "indonesian", "malaysian", "burmese",
    "mexican", "cuban", "puerto rican", "dominican", "haitian", "jamaican",
    "trinidadian", "guatemalan", "honduran", "salvadoran", "nicaraguan",
    "colombian", "venezuelan", "ecuadorian", "peruvian", "bolivian", "chilean",
    "argentinian", "argentine", "brazilian", "panamanian", "costa rican",
    "irish", "scottish", "welsh", "english", "british", "german", "dutch",
    "french", "belgian", "swiss", "austrian", "italian", "sicilian", "spanish",
    "basque", "portuguese", "greek", "turkish", "armenian", "kurdish",
    "polish", "czech", "slovak", "hungarian", "romanian", "bulgarian",
    "serbian", "croatian", "bosnian", "albanian", "slovenian", "macedonian",
    "russian", "ukrainian", "belarusian", "lithuanian", "latvian", "estonian",
    "finnish", "swedish", "norwegian", "danish", "icelandic",
    "nigerian", "ghanaian", "senegalese", "malian", "liberian", "sierra leonean",
    "ethiopian", "eritrean", "somali", "sudanese", "kenyan", "ugandan",
    "tanzanian", "congolese", "cameroonian", "south african", "moroccan",
    "algerian", "tunisian", "egyptian", "libyan",
    "lebanese", "syrian", "palestinian", "jordanian", "iraqi", "iranian",
    "persian", "israeli", "yemeni", "saudi", "afghan", "uzbek", "kazakh",
    "australian", "new zealander", "samoan", "tongan", "fijian", "chamorro",
    # tribal nations named often enough to matter
    "cherokee", "navajo", "dine", "choctaw", "chickasaw", "creek", "muscogee",
    "seminole", "lakota", "dakota", "sioux", "ojibwe", "chippewa", "apache",
    "hopi", "zuni", "pueblo", "iroquois", "haudenosaunee", "mohawk", "seneca",
    "oneida", "shawnee", "lumbee", "catawba", "powhatan", "blackfeet", "crow",
    "cheyenne", "comanche", "kiowa", "osage", "ponca", "pawnee", "shoshone",
    "paiute", "ute", "yakama", "nez perce", "tlingit", "haida", "inuit", "yupik",
}

# A capitalized label: one token, or two joined by a space or hyphen
# ("Scotch-Irish", "African American", "Puerto Rican"). The LABEL group stays
# case-SENSITIVE -- capitalization is most of what makes it a proper ethnonym
# rather than an ordinary adjective -- while every trigger phrase around it is
# wrapped in `(?i:...)`. Getting that split wrong is not cosmetic: with the whole
# pattern case-sensitive, "My people are Scotch-Irish" (sentence-initial, so
# capital M) did not match while "we were part Cherokee" did, and interview_002's
# speaker was labelled Cherokee off a disputed claim about his grandmother's
# mother's side instead of the Scotch-Irish he actually states.
_ETH_LABEL = r"[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)?"
_ETH_QUAL = r"(?i:(?:part|half|mostly|full|pure|quarter)\s+)?"

# (1) first person: "we're Scotch-Irish", "I'm Vietnamese", "My people are
#     Scotch-Irish", "we were part Cherokee"
_ETH_FIRST = re.compile(
    r"(?i:\b(?:I['’]?m|I\s+am|I\s+was|we['’]?re|we\s+are|we\s+were|"
    r"my\s+(?:family|people|folks)\s+(?:is|are|was|were)|"
    r"my\s+(?:mother|father|people)['’]s\s+side\s+(?:is|was))\s+)"
    + _ETH_QUAL + r"(" + _ETH_LABEL + r")\b")
# (2) heritage phrasing, either order: "of Vietnamese descent", "Cherokee blood"
_ETH_HERITAGE = re.compile(
    r"(?i:\b(?:of\s+)?)" + _ETH_QUAL + r"(" + _ETH_LABEL +
    r")(?i:\s+(?:descent|heritage|ancestry|blood|extraction|stock|immigrant|"
    r"immigrants|refugee|refugees)\b)")
# (3) "as a Vietnamese refugee" / "as a Cajun family"
_ETH_AS_A = re.compile(r"(?i:\bas\s+(?:a|an)\s+)" + _ETH_QUAL +
                       r"(" + _ETH_LABEL + r")(?i:\s+\w+)")
# (4) third person, bound to a name: "<Name> was part Cherokee"
_ETH_THIRD = re.compile(r"(?i:^\W{0,3}(?:is|was|were|are)\s+)" + _ETH_QUAL +
                        r"(" + _ETH_LABEL + r")\b")

# Constructions that describe whoever is SPEAKING. Only these are read for the
# interviewee.
_ETH_PATTERNS = (_ETH_FIRST, _ETH_HERITAGE, _ETH_AS_A)
# Constructions that can be bound to a THIRD PARTY by proximity to their name. A
# first-person one cannot: "my Mamaw Opal always claimed we were part Cherokee"
# sits right after Opal's name but the "we" is the speaker's family, not a
# statement about Opal -- reading _ETH_FIRST near a mention attributed the
# speaker's claim to his grandmother.
_ETH_THIRD_PARTY_PATTERNS = (_ETH_HERITAGE,)
# How far after a mention a third-person construction may sit and still be read
# as that person's.
_ETH_NEAR = 60


def normalize_ethnonym(label: str) -> str | None:
    """Canonical lowercase ethnonym, or None when the label is not one we accept."""
    t = re.sub(r"\s+", " ", str(label or "").strip().lower()).strip(".,;:")
    if not t:
        return None
    if t in ETHNONYMS:
        return t
    # tolerate the hyphen/space variants of the same label
    for alt in (t.replace("-", " "), t.replace(" ", "-")):
        if alt in ETHNONYMS:
            return alt
    return None


# Verbs and framings that ATTRIBUTE a heritage claim to somebody else, or hold it at
# arm's length, rather than stating it. Interview_002 is the case: "My people are
# Scotch-Irish, mostly, ... though my Mamaw Opal always CLAIMED we were part Cherokee
# on her mother's side. Nobody ever proved it and nobody ever disproved it." Both
# constructions match, so the rule saw two self-applied labels and abstained -- which
# left the speaker's ethnicity to whatever the model happened to return that run, on a
# field that feeds surrogate NAME selection. The speaker states one of these two and
# reports the other as family lore, and that difference is written in the text.
_ETH_HEDGE = re.compile(
    r"\b(?:claim(?:s|ed|ing)?|said|says|saying|reckon(?:s|ed)?|suppose(?:d|dly)?|"
    r"supposably|allege(?:s|d)?|thought|believe[ds]?|heard\s+tell|story\s+goes|"
    r"family\s+lore|legend|rumor(?:ed)?|myth|maybe|might\s+be|could\s+be|"
    r"never\s+(?:proved|proven)|not\s+sure)\b", re.I)
# How far before a claim an attribution verb may sit and still frame it.
_ETH_HEDGE_WINDOW = 90


def ethnicity_claims_with_pos(text: str, patterns=None):
    """Every (canonical_label, evidence_quote, start_offset) an accepted construction
    yields. The offset is what lets a caller ask whether the claim is HEDGED."""
    out = []
    for rx in (patterns or _ETH_PATTERNS):
        for m in rx.finditer(text):
            canon = normalize_ethnonym(m.group(1))
            if canon:
                out.append((canon, m.group(0).strip(), m.start()))
    return out


def ethnicity_claims(text: str, patterns=None):
    """Every (canonical_label, evidence_quote) an accepted construction yields.

    `patterns` defaults to the first-person/self-description set. Pass
    `_ETH_THIRD_PARTY_PATTERNS` when binding a claim to somebody else's name.
    """
    return [(c, ev) for c, ev, _pos in ethnicity_claims_with_pos(text, patterns)]


def unhedged_ethnicity_claims(text: str, patterns=None):
    """The claims that are STATED rather than attributed to somebody or hedged.

    Used only to break a tie: when several labels are self-applied, a single unhedged
    one is the speaker's own statement about themselves and the rest are family lore.
    With none or several unhedged, the rule still abstains.
    """
    out = []
    for canon, ev, pos in ethnicity_claims_with_pos(text, patterns):
        before = text[max(0, pos - _ETH_HEDGE_WINDOW):pos]
        if not _ETH_HEDGE.search(before):
            out.append((canon, ev))
    return out


def infer_ethnicity(transcript: str, person_entities: list[Entity],
                    interviewee: Entity) -> None:
    """RULE layer for `ethnicity`.

    The interviewee's own label is read from first-person constructions in the
    SUBJECT'S TURNS ONLY, so the interviewer's "my family's Irish" cannot become
    the subject's. A named person's label must come from a construction bound to
    one of THEIR mentions -- which is what keeps the speaker's ethnicity from
    silently spreading to everyone they talk about.
    """
    spoken = mask_to_subject(transcript)
    if not interviewee.attributes.get("ethnicity"):
        claims = ethnicity_claims(spoken)
        labels = {c for c, _ev in claims}
        if len(labels) > 1:
            # Several labels self-applied. Before abstaining, drop the ones the
            # speaker ATTRIBUTES to somebody else or hedges ("my Mamaw always claimed
            # we were part Cherokee"); if exactly one plainly-stated label remains,
            # that is the speaker's own. See `_ETH_HEDGE`.
            unhedged = unhedged_ethnicity_claims(spoken)
            ulabels = {c for c, _ev in unhedged}
            if len(ulabels) == 1:
                claims, labels = unhedged, ulabels
        if len(labels) == 1:
            canon, ev = claims[0]
            interviewee.attributes["ethnicity"] = canon
            interviewee.attributes.setdefault("ethnicity_evidence", ev)
            interviewee.attributes.setdefault("ethnicity_basis", "stated")
        elif len(labels) > 1:
            interviewee.flag_entity(
                "several ethnicity/heritage labels are self-applied in the "
                "subject's speech; left unset for the LLM / a human to resolve")

    for ent in person_entities:
        if ent is interviewee or ent.attributes.get("ethnicity"):
            continue
        found = set()
        ev_for = {}
        for m in ent.mentions:
            after = transcript[m.end:m.end + _ETH_NEAR]
            hit = _ETH_THIRD.match(after)
            if hit:
                canon = normalize_ethnonym(hit.group(1))
                if canon:
                    found.add(canon)
                    ev_for.setdefault(canon, (ent.sorted_mentions[0] + hit.group(0)).strip())
            for canon, ev in ethnicity_claims(after, _ETH_THIRD_PARTY_PATTERNS):
                found.add(canon)
                ev_for.setdefault(canon, ev)
        if len(found) == 1:
            canon = found.pop()
            ent.attributes["ethnicity"] = canon
            ent.attributes.setdefault("ethnicity_evidence", ev_for.get(canon, ""))
            ent.attributes.setdefault("ethnicity_basis", "stated")


def _personal_signal(transcript: str, ent, kin_ids: set) -> bool:
    """True when a name that matches the public-figure list is, in THIS transcript,
    a private individual (a relative, someone with a professional role in the
    interviewee's life, or a namesake introduced personally) -> must be redacted."""
    if ent.entity_id in kin_ids:                          # the interviewee's relative
        return True
    for m in ent.mentions:
        pre = transcript[max(0, m.start - 24):m.start]
        if _POSS_NAME.search(pre) or _NAMING.search(pre):
            return True
        if PROFESSIONAL_CONTEXT.search(transcript[max(0, m.start - 60):m.end + 60]):
            return True
        if _NOT_FAMOUS.match(transcript[m.end:m.end + 40]):
            return True
    return False


"""
Infer high level attributes about PERSON entities. Earlier, we've detected PERSON
entities, clustered mentions, and built RELATED_TO edges and etc. 
"""
def infer_person_attributes(
    transcript: str, person_entities: list[Entity], edges: list[Edge]
) -> None:
    kin_targets = {e.target for e in edges if e.relation == Relation.RELATED_TO}
    kin_sources = {e.source for e in edges if e.relation == Relation.RELATED_TO}
    kin_ids = kin_targets | kin_sources

    for ent in person_entities:
        attrs = ent.attributes

        # longest name form usually contains the most complete name
        longest = ent.sorted_mentions[0] if ent.sorted_mentions else ""

        # THE shared split (graph/merge_strings.py): titles and kinship words are
        # stripped, the first/last real tokens become given/surname, and a lone
        # token left behind by an HONORIFIC is slotted as a SURNAME rather than a
        # given name. This function used to carry its own copy of the split, which
        # always slotted a lone token as the given name -- so "Father Nguyen" set
        # `given_name="Nguyen"`.
        gn, sn = split_name_parts(longest)
        if gn or sn:
            attrs.setdefault("given_name", gn)
            attrs.setdefault("surname", sn)

        # NOW WE DEAL WITH SPECIAL TYPES OF PEOPLE
        forms_lower = {f.lower() for f in ent.sorted_mentions}  # lowercase everything

        if forms_lower & PUBLIC_FIGURES:
            # Default for a listed famous name is to KEEP it (it reveals nothing
            # about the interviewee). Redact only when a personal signal says this
            # is a private namesake -- leaking a private person is the unacceptable
            # error; over-redacting a celebrity is the safe one.
            #
            # This keep is PROVISIONAL: the closed PUBLIC_FIGURES list is not the
            # sole authority. `replace` is arbitrated by graph/second_line.py with
            # conflict_policy=safe_direction and unsafe=False, so the keep survives
            # only if the LLM also affirms a public figure AND the checkers in
            # graph/checks/persons.py find no personal signal; any disagreement
            # resolves toward more redaction. Rules-only behavior here is unchanged.
            if not _personal_signal(transcript, ent, kin_ids):
                ent.subtype = "PUBLIC_FIGURE"
                attrs["replace"] = False
                continue
            ent.flag_entity(
                "name matches a public figure but is used personally "
                "(relative / professional / namesake); treated as private"
            )

        if ent.entity_id in kin_targets or ent.entity_id in kin_sources:
            ent.subtype = ent.subtype or "FAMILY"
        else:
            # professional-context words within 60 chars of any mention?
            for m in ent.mentions:
                window = transcript[max(0, m.start - 60):m.end + 60]
                if PROFESSIONAL_CONTEXT.search(window):
                    ent.subtype = "PROFESSIONAL"
                    break
        attrs.setdefault("replace", True)

        # Kinship.py may have set the gender already; also pronouns could refine it
        # once coref runs. Leave None otherwise for gender-neutral substitutions.
        attrs.setdefault("gender", None)