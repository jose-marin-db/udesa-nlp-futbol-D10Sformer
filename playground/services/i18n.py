"""Spanish ↔ English team names for the playground."""

from __future__ import annotations

from simulation.bracket import SPANISH_TO_ENGLISH

ENGLISH_TO_SPANISH = {v: k for k, v in SPANISH_TO_ENGLISH.items()}


def to_english(name: str) -> str:
    return SPANISH_TO_ENGLISH.get(name, name)


def to_spanish(name_en: str) -> str:
    return ENGLISH_TO_SPANISH.get(name_en, name_en)
