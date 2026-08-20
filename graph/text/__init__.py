"""
`graph.text` -- transcript utilities with no knowledge of the graph.

Package marker. Two leaf modules, both pure functions over text and offsets:

    sentences.py  abbreviation-aware sentence spans
    turns.py      speaker-turn segmentation and subject masking

Kept separate because "where does a sentence end?" and "who was talking?" are
questions the rules, the checkers and the LLM layer all need answered the same
way.
"""
