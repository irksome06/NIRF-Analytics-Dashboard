"""Conversation context: "what about engineering?", "and for 2022?", "show more", "same for pharmacy", "include band estimates"."""
from __future__ import annotations

import re

from engine.registry import QUESTIONS
from nlp.intent import THRESHOLD

SWITCH = re.compile(r"^(?:what about|how about|same|now|instead|what if|make it|switch|try)\b")
MARKER = re.compile(r"^(?:and|also|what about|how about|same|now|then|instead|but|what if|try|switch|only|make it|do it|repeat|again|and for|for)\b")
MORE = re.compile(r"\b(?:show|give|see)? ?(?:me )?(?:more|a few more|more results)\b|^top \d+$")
LESS = re.compile(r"\b(?:fewer|less|show less|shorter)\b")


def inherit(last: dict) -> dict:
    """Parameters of the previous (validated) query, with year fields made consistent for any next question."""
    out = {k: v for k, v in last.items() if k != "question" and v is not None}
    yf, yt, ya, yb, y = (last.get(k) for k in ("year_from", "year_to", "year_a", "year_b", "year"))
    if yf and yt:
        out.update(year=yt, year_a=ya or yf, year_b=yb or yt)
    elif ya and yb:
        out.update(year=yb, year_from=ya, year_to=yb)
    elif y:
        out.update(year_to=y)
    return out


def resolve(t: str, ents: dict, qid: str | None, score: float, last: dict | None) -> tuple[str | None, dict, bool]:
    """Return (question id, entities, was_followup)."""
    if not last or last.get("question") not in QUESTIONS:
        return qid, ents, False
    lq = last["question"]
    params = QUESTIONS[lq].params
    base = inherit(last)
    if "top_n" in params and (MORE.search(t) or LESS.search(t)) and not ents.get("top_n") and len(t.split()) <= 5:
        n = int(last.get("top_n") or 10)
        return lq, {**base, "top_n": min(30, n + 10) if MORE.search(t) else max(3, n - 5)}, True
    marker = bool(MARKER.match(t))
    short = len(t.split()) <= 4
    data_ents = {k: v for k, v in ents.items() if k in ("category", "year", "year_from", "year_to", "year_a", "year_b", "state", "state_a", "state_b", "institution",
                                                         "parameter", "x", "y", "top_n", "exact_only", "bottom")}
    if data_ents and SWITCH.match(t) and len(t.split()) <= 8:
        return lq, {**base, **ents}, True          # "what about IIT Delhi?", "same for pharmacy": always the previous question
    if data_ents and (qid is None or score < THRESHOLD + 3) and (marker or short):
        return lq, {**base, **ents}, True          # only new filters were given: re-run the previous question with them
    if qid is not None and marker:
        return qid, {**base, **ents}, True          # "and the biggest decliners?": new question, same filters
    return qid, ents, False
