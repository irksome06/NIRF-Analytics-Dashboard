"""Soft intent signal: stem + synonym-class overlap between the question text and each analysis' description (IDF-weighted cosine).
It never decides alone; it nudges the cue/phrase score so unseen paraphrases still land on the right analysis."""
from __future__ import annotations

import math
import re
from functools import lru_cache

from engine.registry import QUESTIONS
from nlp.preprocess import normalise

CLASSES = {
    "UP": "improv climb ris rise gain jump advanc upward better boost progress higher increas grow growth surge",
    "DOWN": "declin drop fall fell slip decreas lost lose worse down downward decay sink plung deteriorat lower shrink",
    "STEADY": "consistent always permanent regular stable steady perennial throughout every remain stay stayed retain unchanged same",
    "SPREAD": "spread disper variab vary vari variation differ range widely wide variance deviation",
    "RELATE": "correlat relat link associat affect drive influenc depend impact relationship connect",
    "ODD": "unusual abnormal outlier anomal exception suspicious odd extreme unexpected remarkabl surprising out-of-line",
    "SHARE": "share fraction proportion percent percentage ratio",
    "QUART": "quartile quarter quarters fourth",
    "NEW": "new newly fresh newcomer entrant entry entries appear debut first-time",
    "LEAVE": "exit exited left leave leaving dropped-out out removed",
    "SCORE": "score scores scoring mark marks points",
    "RANK": "rank ranks ranking rankings position positions place places",
    "STATE": "state states region regions regional geograph geography",
    "TOP": "top best leading highest toppers topper leader leaders",
    "LOW": "low lowest poor weak worst bottom least",
    "COMPARE": "compare compared comparison versus vs against between",
    "TIME": "time year years yearly annual evolv evolution trend trends history",
}
_STEM = re.compile(r"(?:ing|ed|es|s|ly|ion|ions)$")
STOP = set("the a an of in on for to from by with and or is are was were be been it its their them that this these those what which who how do does did has have had me show give tell please all any can you i".split())


def stem(w: str) -> str:
    return _STEM.sub("", w) if len(w) > 4 else w


@lru_cache(maxsize=1)
def _index():
    cmap = {}
    for cid, words in CLASSES.items():
        for w in words.split():
            cmap[stem(w)] = cid
            cmap[w] = cid

    def concepts(text: str) -> set[str]:
        out = set()
        for w in normalise(text).split():
            if w in STOP or w.isdigit():
                continue
            s = stem(w)
            out.add(cmap.get(w) or cmap.get(s) or s)
        return out

    profiles = {qid: concepts(q.title + " " + " ".join(q.keywords)) for qid, q in QUESTIONS.items()}
    df = {}
    for c in set().union(*profiles.values()):
        df[c] = sum(c in p for p in profiles.values())
    n = len(profiles)
    idf = {c: math.log((n + 1) / (d + 0.5)) for c, d in df.items()}
    norms = {qid: math.sqrt(sum(idf[c] ** 2 for c in p)) or 1.0 for qid, p in profiles.items()}
    return concepts, profiles, idf, norms


def similarity(text: str) -> dict[str, float]:
    """qid -> cosine similarity in [0, 1] between the text and each analysis profile."""
    concepts, profiles, idf, norms = _index()
    q = {c for c in concepts(text) if c in idf}
    if not q:
        return {k: 0.0 for k in profiles}
    qn = math.sqrt(sum(idf[c] ** 2 for c in q)) or 1.0
    return {qid: sum(idf[c] for c in q & p) / (qn * norms[qid]) for qid, p in profiles.items()}
