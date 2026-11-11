"""M1 - Intent classifier: maps a question to a registered analysis.

Score = registry keyword phrases + weighted cue patterns (nlp/cues.py) + entity-driven rules
(institution / two institutions / leaderboard / states ...). Returns (best id | None, score, top-3)."""
from __future__ import annotations

import re

from engine.registry import QUESTIONS
from nlp.cues import CUES, DOWN, UP
from nlp.preprocess import normalise
from nlp.semantic import similarity

THRESHOLD = 3.0
STOP = {"the", "of", "in", "a", "and", "to", "from", "which", "how", "what", "have", "has", "is", "are", "their", "than", "over", "across", "for", "with", "or", "an", "be"}
LEAD_CUE = r"\b(?:top|best|highest ranked|leading|first|ranked first|number 1|number one|bottom|worst|lowest ranked|topper|toppers)\b"
NOUN = r"\b(?:institutions?|colleges?|universit(?:y|ies)|institutes?|schools?)\b"
NOT_LEADERBOARD = (r"improv|declin|consisten|always|enter|exit|left|joined|state|compare|map|distribution|distributed|spread|geograph|correlat|volatil|outlier|gap|band|moved|"
                   r"quartile|percent|new entr|trend|over time|parameter|share|average|catching|most years|unstable|despite|rest|remaining|jump|drop|fell|climb|rose|region|getting more|than before")


def _phrase_score(t: str, phrase: str) -> float:
    p = normalise(phrase)
    return 2.0 + len(p.split()) if p and p in t else 0.0


def classify(text: str, entities: dict | None = None) -> tuple[str | None, float, list[tuple[str, float]]]:
    """Return (best question id or None, score, ranked candidates)."""
    t = normalise(text)
    ents = entities or {}
    toks = set(t.split()) - STOP
    scores: dict[str, float] = {}
    cue: dict[str, float] = {}
    for qid, q in QUESTIONS.items():
        cue[qid] = sum(w for pat, w in CUES.get(qid, ()) if re.search(pat, t))
        scores[qid] = sum(_phrase_score(t, k) for k in q.keywords) + 0.35 * len(toks & (set(normalise(q.title).split()) - STOP)) + cue[qid]
    sim = similarity(t)
    for qid in scores:
        scores[qid] += 5.0 * sim[qid]            # soft signal for unseen paraphrases
    # a specific analysis beats the broad one it is a refinement of ("states ... catching up" vs "states ... top 100")
    if any(cue[q] >= 6 for q in ("2.2", "2.3", "2.4", "2.5", "6.5")):
        scores["2.1"] -= 5
    if cue["6.5"] >= 6 or cue["3.5"] >= 6:
        scores["3.1"] -= 5

    has_up, has_down = bool(re.search(UP, t)), bool(re.search(DOWN, t))
    if has_up and has_down:                       # "improved ... despite declining", "jumps and drops": not a plain 1.2 / 1.3
        scores["1.2"] -= 3; scores["1.3"] -= 3

    if cue["6.1"] == 0 and re.search(r"\b(?:new|enter\w*|exit\w*|left|joined|came|fresh|newcomers?)\b", t):
        scores["6.1"] -= 4                      # "new names in the top 10 compared with last year" is 7.2, not top-10-vs-rest
    if not re.search(r"\b(?:how many|number of|count|stayed|same|unchanged|each year|every year|proportion|percent\w*|put)\b", t):
        scores["7.5"] -= 3                      # 7.5 is a COUNT of movers; plain "moved up the most" is 1.2

    # ---- institution-driven routing
    has_inst, two_inst = "institution" in ents, "institution_b" in ents
    ranged = "year_from" in ents
    trendish = bool(re.search(r"\b(?:over the years|over time|history|trend|changed?|journey|trajectory|progress\w*|since|gone up|went up|going)\b", t)) or ranged
    if two_inst:
        scores["1.7"] += 10 if re.search(r"\b(?:compare\w*|vs|versus|against|than|better|difference|between|and)\b", t) else 8
        for q in scores:
            if q not in ("1.7",):
                scores[q] -= 4
    elif has_inst:
        snapshot_words = r"\b(?:profile|snapshot|overview|scorecard|summary|details?|info|information|about|facts|stats|statistics|how good|how strong|parameters?|strengths?|weakness\w*|scores?|score across|rank|do|did|get|got|stand|performed?)\b"
        if ("year" in ents and not ranged and re.search(snapshot_words, t) and not re.search(r"\b(?:over the years|over time|history|trend|since)\b", t)) \
                or (re.search(r"\b(?:profile|snapshot|overview|scorecard|summary|details?|information|about|facts|stats|statistics|how good|how strong|parameters?|strengths?|weakness\w*|scores? across)\b", t) and not trendish):
            scores["1.8"] += 9
        elif re.search(r"\b(?:rank\w*|since|over|history|trend|changed?|perform\w*|progress\w*|gone|journey)\b", t) or ranged:
            scores["1.1"] += 8
        else:
            scores["1.1"] += 3
        for q in scores:
            if q not in ("1.1", "1.8", "1.7"):
                scores[q] -= 3

    # ---- other entity / phrase rules
    if ents.get("state") and "state_b" not in ents and not has_inst and re.search(r"\b(?:presence|representation|footprint|foothold|share)\b", t):
        scores["2.3"] += 8
    if "x" in ents and ents.get("y") == "score" and ents.get("x") not in ("RPC", "PR") and re.search(r"\b(?:relat\w*|correlat\w*|link\w*|associat\w*|affect\w*|driv\w*|influenc\w*|depend\w*|impact\w*)\b", t):
        scores["4.6"] += 7                      # "is graduation outcomes related to overall score": one parameter vs the score
    if "state_b" in ents and re.search(r"\b(compare|versus|vs|against|with|and|or|better|than)\b", t):
        scores["2.4"] += 4
    if re.search(r"\b(map|geograph|on a map)\b", t) and not re.search(r"distribution of top|spread|chang", t):
        scores["2.6"] += 3
    two_vars = "x" in ents and "y" in ents and ents["y"] != "score" or ("x" in ents and re.search(r"\b(scatter|relationship|relate|related|correlat|versus|vs|against)\b", t))
    if two_vars and re.search(r"\b(scatter|relationship|relate|related|vs|versus|against|between|correlat|plot)\b", t) and not has_inst:
        scores["4.6"] += 2.5
        if re.search(r"\b(?:plot|scatter)\b.*\b(?:against|vs|versus)\b", t):
            scores["4.6"] += 3
    if re.search(r"\b(top 10|top ten)\b", t) and re.search(r"\b(enter|exit|left|joined|entered|exited)\w*\b", t):
        scores["7.2"] += 4
    if re.search(r"\b(consistent|consistently|always|every year|multiple years|most years)\b", t) and re.search(r"top 10", t):
        scores["1.5"] += 4
    if re.search(r"\b(declin|drop|fell|fall|worst)\w*\b", t) and re.search(r"\b(most|largest|biggest|greatest)\b", t) and "year_a" not in ents and not has_up:
        scores["1.3"] += 2

    # ---- leaderboard: "top 10 engineering colleges in 2024", "best universities", "who is ranked first in law", "bottom 10 ..."
    leaderboard = re.search(LEAD_CUE, t) and (re.search(NOUN, t) or re.search(r"\b(?:ranked first|number 1|number one|topper|toppers)\b", t) or "category" in ents)
    if leaderboard and not re.search(NOT_LEADERBOARD, t) and not has_inst:
        scores["1.6"] += 6
    if re.search(r"\b(?:bottom|worst|lowest ranked)\b", t) and re.search(NOUN, t) and not re.search(NOT_LEADERBOARD.replace("|worst", ""), t) and not has_inst:
        scores["1.6"] += 4

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, score = ranked[0]
    if score < THRESHOLD:
        return None, score, ranked[:3]
    return best, score, ranked[:3]


def intent_of(qid: str) -> str:
    return QUESTIONS[qid].intent
