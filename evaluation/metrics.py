"""
Two primitives shared by scoring and reporting.

They live here so the two layers do not have to import each other: an accuracy
ratio, and the stand-in entity used when a gold row has no matching entity.
"""

from __future__ import annotations


def _acc(num, den):
    return (num / den) if den else None


class _NoEnt:
    """Stand-in so a missing entity reads as 'untyped' instead of raising."""
    subtype = None
