"""
Relationship vocabulary shared by the relation checker and the FAMILY-subtype
inference.

PURPOSE
    One frozen list of the kin words that count as a family relationship, so the
    relation verifier and the subtype/role checkers accept exactly the same
    vocabulary.

FIT
    Imported by `graph/checks/relation_evidence.py` (where it is unioned with a
    social-tie set to form `_REL_WORDS`) and by `graph/checks/persons.py`. No
    dependencies of its own.

Kept as a literal set rather than derived from `kinship.KINSHIP_GENDER`: that table
carries hyphenated variants and loose terms ("ex", "relative", "in-law") that widen
the accepted vocabulary and would change which proposals clear the checker. This is
the exact set the verifier was tuned against.
"""

KIN_WORDS = {
    "mother", "mom", "father", "dad", "parent", "son", "daughter", "child",
    "kid", "brother", "sister", "sibling", "aunt", "uncle", "cousin", "niece",
    "nephew", "grandmother", "grandfather", "grandma", "grandpa", "grandson",
    "granddaughter", "wife", "husband", "spouse", "partner", "in-law",
    "mother-in-law", "father-in-law", "brother-in-law", "sister-in-law",
    "stepmother", "stepfather", "stepbrother", "stepsister", "half-brother",
    "half-sister", "godmother", "godfather", "grandparent", "grandchild",
}
