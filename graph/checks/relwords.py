"""
Relationship vocabulary shared by the relation checker and the FAMILY-subtype
inference.

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
