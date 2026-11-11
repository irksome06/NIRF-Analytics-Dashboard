"""Vocabulary and spelling correction for the Ask NIRF chatbot.

Unknown words are corrected against a vocabulary built from the question registry, category / state / parameter terms
and every word that occurs in an institution name (so "madrass" -> "madras", "enginering" -> "engineering")."""
from __future__ import annotations

import difflib
import re

import pandas as pd

from engine.registry import QUESTIONS
from nlp.preprocess import normalise

# everyday words that must never be "corrected" into domain words
COMMON = set("""a an and or the of in on for to from by with without about across between among over under into than then that this these those what which who whom whose
how why when where is are was were be been being do does did has have had can could should would will may might must show give tell list plot draw chart see get find
me my our us we you your it its their them they he she him her not no yes also more most less least many much few some any all each every other another such same
done doing going gone made make making look looking want need getting seen shown name names old high low big large small long short good great
top best worst first last next previous latest recent current year years rank ranks ranking rankings score scores institute institution institutions college colleges
university universities state states india indian national list compare change changed changes trend vs versus whether does""".split())

_CACHE: dict[int, tuple] = {}


def _words(texts) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out.update(w for w in normalise(str(t)).split() if len(w) >= 3 and not w.isdigit())
    return out


def vocabulary(df: pd.DataFrame) -> tuple[set[str], list[str]]:
    """(all known words, words eligible as correction targets). Cached per dataframe object."""
    key = id(df)
    if key not in _CACHE:
        from nlp.entities import CATEGORY_TERMS, PARAM_TERMS, STATE_ALIASES
        domain = _words([q.title for q in QUESTIONS.values()] + [k for q in QUESTIONS.values() for k in q.keywords])
        domain |= _words([t for t, _ in CATEGORY_TERMS] + [t for t, _ in PARAM_TERMS] + list(STATE_ALIASES))
        domain |= _words(df["state"].dropna().unique())
        names = _words(df.drop_duplicates("institution_key")["institute_name"])
        # extra domain words users type that the registry titles do not contain
        domain |= {"improved", "declined", "distribution", "correlation", "quartile", "histogram", "scatter", "volatility", "perception",
                   "engineering", "pharmacy", "management", "architecture", "agriculture", "medical", "dental", "ranking", "rankings"}
        from engine.data_loader import ALIASES
        known = domain | names | COMMON | set(ALIASES)
        targets = sorted(domain | names)
        _CACHE.clear()
        _CACHE[key] = (known, targets)
    return _CACHE[key]


def spell_fix(text: str, df: pd.DataFrame) -> tuple[str, dict[str, str]]:
    """Return (normalised text with unknown words corrected, {wrong: right})."""
    known, targets = vocabulary(df)
    out, fixes = [], {}
    for tok in normalise(text).split():
        if tok in known or tok.isdigit() or len(tok) < 5 or re.search(r"\d", tok):
            out.append(tok)
            continue
        hit = [h for h in difflib.get_close_matches(tok, targets, n=3, cutoff=0.85) if len(h) >= 4][:1]
        if hit:
            fixes[tok] = hit[0]
            out.append(hit[0])
        else:
            out.append(tok)
    return " ".join(out), fixes
