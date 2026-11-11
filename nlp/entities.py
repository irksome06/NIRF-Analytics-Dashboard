"""M1 - Entity extraction: years, category, states, institutions, parameters, counts."""
from __future__ import annotations

import re

import pandas as pd

from engine.data_loader import ALIASES, PARAMS, resolve_institution, state_options
from nlp.preprocess import normalise

YEAR_MIN, YEAR_MAX = 2017, 2025

CATEGORY_TERMS = [  # longest / most specific first
    ("state public university", "State Public University"), ("state public universities", "State Public University"),
    ("open university", "Open University"), ("open universities", "Open University"),
    ("skill university", "Skill University"), ("skill universities", "Skill University"),
    ("engineering", "Engineering"), ("engineering colleges", "Engineering"), ("btech", "Engineering"), ("b tech", "Engineering"),
    ("management", "Management"), ("mba", "Management"), ("b-school", "Management"), ("business school", "Management"),
    ("pharmacy", "Pharmacy"), ("pharma", "Pharmacy"), ("medical", "Medical"), ("medicine", "Medical"), ("mbbs", "Medical"),
    ("dental", "Dental"), ("law", "Law"), ("legal", "Law"), ("architecture", "Architecture"), ("planning", "Architecture"),
    ("agriculture", "Agriculture"), ("agricultural", "Agriculture"), ("innovation", "Innovation"),
    ("research institutions", "Research"), ("research institutes", "Research"), ("research category", "Research"),
    ("universities", "University"), ("university category", "University"), ("colleges", "College"), ("college category", "College"),
    ("overall", "Overall"),
]
STATE_ALIASES = {"tn": "Tamil Nadu", "tamilnadu": "Tamil Nadu", "ka": "Karnataka", "up": "Uttar Pradesh", "mp": "Madhya Pradesh",
                 "ap": "Andhra Pradesh", "wb": "West Bengal", "mh": "Maharashtra", "j&k": "Jammu and Kashmir", "jammu kashmir": "Jammu and Kashmir",
                 "kerela": "Kerala", "orissa": "Odisha", "pondicherry": "Puducherry", "bengal": "West Bengal", "delhi ncr": "Delhi", "new delhi": "Delhi"}
PARAM_TERMS = [("graduation outcomes", "GO"), ("graduation outcome", "GO"), ("graduation", "GO"), ("tlr", "TLR"), ("teaching", "TLR"),
               ("rpc", "RPC"), ("professional practice", "RPC"), ("research", "RPC"), ("outreach", "OI"), ("inclusivity", "OI"),
               ("inclusiveness", "OI"), ("oi", "OI"), ("perception", "PR"), ("pr", "PR")]
EVERYDAY = {"down", "up", "gone", "since", "over", "year", "years", "rank", "most", "best", "good", "new", "old", "high", "low", "large", "small", "top", "first", "last", "next", "change", "changed", "score", "scores", "much", "many", "more", "less", "perform", "performed", "did", "get", "got", "profile"}
CATEGORY_WORDS = {"resources", "resource", "learning", "outcomes", "outcome", "professional", "practice", "outreach", "inclusivity", "perception", "graduation", "teaching", "law", "school", "schools", "pharmacy", "dental", "medical", "agriculture", "architecture", "research", "innovation", "engineering", "management", "college", "colleges"}
GENERIC = {"institution", "institutions", "university", "universities", "college", "colleges", "institute", "institutes", "top", "rank", "ranks",
           "ranking", "rankings", "score", "scores", "state", "states", "india", "indian", "national", "technology", "engineering", "management",
           "medical", "overall", "research", "teaching", "perception", "year", "years", "the", "of", "in", "and", "for", "from", "to", "between",
           "how", "what", "which", "has", "have", "with", "by", "most", "change", "changed", "compare", "show", "me", "trend", "over", "time",
           "average", "median", "distribution", "correlation", "outreach", "inclusivity", "graduation", "outcomes", "parameter", "parameters",
           "improved", "declined", "category", "categories", "nirf", "vs", "versus", "than", "is", "are", "a", "an", "all", "map", "number",
           "new", "top", "best", "worst", "high", "low", "strong", "weak", "large", "largest", "small", "unusually", "volatility", "gap", "gaps",
           "quartile", "quartiles", "band", "bands", "science", "sciences", "education", "law", "pharmacy", "dental", "agriculture", "architecture"}


_Y = r"(20(?:1[7-9]|2[0-5]))"


def _years(t: str) -> dict:
    ys = [int(y) for y in re.findall(rf"\b{_Y}\b", t)]
    out: dict = {}
    # short end year in a range: "2018-22", "2018 to 22"
    m = re.search(rf"\b{_Y}\s*(?:-|to|until|till)\s*(\d{{2}})\b(?!\d)", t)
    if m and YEAR_MIN <= 2000 + int(m.group(2)) <= YEAR_MAX:
        ys = [int(m.group(1)), 2000 + int(m.group(2))]
    # "last 5 years", "past three editions", "recent years", "last decade"
    rel = re.search(r"\b(?:last|past|previous|recent) (\d{1,2}) (?:years?|editions?|rankings?)\b", t)
    n = int(rel.group(1)) if rel else 3 if re.search(r"\brecent years\b", t) else (YEAR_MAX - YEAR_MIN + 1) if re.search(r"\b(?:last|past) decade\b", t) else None
    if n and not ys:
        a = max(YEAR_MIN, YEAR_MAX - n + 1)
        return dict(year_from=a, year_to=YEAR_MAX, year_a=a, year_b=YEAR_MAX, year=YEAR_MAX)
    if re.search(r"\b(?:latest|current|most recent|this year|last year|this edition|now|today)\b", t) and not ys:
        ys = [YEAR_MAX]
    since = re.search(rf"\bsince {_Y}\b", t) or re.search(rf"\b{_Y}\b.*\b(?:to|until|till|through) (?:now|today|present|date|latest)\b", t) \
        or re.search(rf"\b(?:from|starting|beginning) {_Y}\b(?!.*\b(?:to|until|till|through|and)\b)", t)
    if since:
        a = int(since.group(1))
        return dict(year_from=a, year_to=YEAR_MAX, year=YEAR_MAX, year_a=a, year_b=YEAR_MAX)
    if len(ys) >= 2:
        a, b = min(ys), max(ys)
        out.update(year_from=a, year_to=b, year_a=a, year_b=b, year=b)
    elif len(ys) == 1:
        y = ys[0]
        out.update(year=y, year_to=y, year_b=y)
        out["year_a"] = y - 1 if y > YEAR_MIN else y
    return out


def _category(t: str) -> str | None:
    for term, cat in CATEGORY_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", t):
            return cat
    return None


def _states(t: str, df: pd.DataFrame, raw: str = "") -> list[str]:
    found = []
    for s in state_options(df):
        i = t.find(s.lower())
        if i >= 0:
            found.append((i, s))
    for alias, s in STATE_ALIASES.items():
        if len(alias) <= 2 and not re.search(rf"\b{alias.upper()}\b", raw):   # "up", "mp", "ka" are everyday words: only count them when typed in capitals
            continue
        m = re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", t)
        if m:
            found.append((m.start(), s))
    seen, out = set(), []
    for _, s in sorted(found):
        if s not in seen:
            seen.add(s)
            out.append(s)
    # 'delhi' also matches 'new delhi'; 'maharashtra' etc. are unique already
    return out


def _params(t: str) -> list[str]:
    hits = []
    for m in re.finditer(r"\b(?:against|vs|versus|and|with|between|plot|scatter|of)\s+go\b|\bgo\s+(?:against|vs|versus|and|with)\b", t):
        hits.append((m.start(), "GO"))
    for term, p in PARAM_TERMS:
        for m in re.finditer(rf"(?<![a-z]){re.escape(term)}(?![a-z])", t):
            hits.append((m.start(), p))
    hits.sort()
    out = []
    for _, p in hits:
        if p not in out:
            out.append(p)
    return out


def _top_n(t: str) -> int | None:
    for pat in (r"\b(?:top|best|worst|bottom|first|leading|lowest) (\d{1,2})\b(?! ?(?:band|percent|%|\d))", r"\b(\d{1,2}) (?:best|top|worst|bottom|leading|lowest)\b"):
        m = re.search(pat, t)
        if m and not re.search(r"\b(in|into|within|of|from) the top " + m.group(1), t):
            n = int(m.group(1))
            return n if 3 <= n <= 30 else None
    m = re.search(r"\b(\d{1,2}) (?:institutions|universities|colleges|institutes|states)\b", t)
    if m and 3 <= int(m.group(1)) <= 30:
        return int(m.group(1))
    return None


def _institutions(t: str, pool: pd.DataFrame) -> tuple[list[str], str | None]:
    """Return (unique institution names found, ambiguous text if any)."""
    toks = t.split()
    state_texts = {a.lower() for a in STATE_ALIASES} | {x.lower() for x in pool["state"].dropna().unique()}
    found, ambiguous, used = [], None, set()
    for n in range(min(6, len(toks)), 0, -1):
        for i in range(len(toks) - n + 1):
            if any(j in used for j in range(i, i + n)):
                continue
            win = toks[i:i + n]
            if all(w in GENERIC or w in EVERYDAY or w in CATEGORY_WORDS for w in win):
                continue
            if n == 1 and (win[0] in EVERYDAY or win[0] in GENERIC or win[0].isdigit() or len(win[0]) < 3 or win[0] in {s.lower() for s in STATE_ALIASES}):
                continue
            text = " ".join(win)
            if text in state_texts:
                continue
            hit = resolve_institution(pool, text)
            if len(hit) == 1:
                found.append(hit["institute_name"].iloc[0])
                used.update(range(i, i + n))
            elif 1 < len(hit) <= 25 and ambiguous is None and (n >= 2 or win[0] in ALIASES):
                ambiguous = text
    return found, ambiguous


def extract(text: str, df: pd.DataFrame, pool: pd.DataFrame | None = None, raw: str | None = None) -> dict:
    """Return a dict of every entity found; missing entities are simply absent."""
    t = normalise(text)
    pool = pool if pool is not None else df.drop_duplicates("institution_key")
    out: dict = {}
    out.update(_years(t))
    cat = _category(t)
    if cat:
        out["category"] = cat
    states = _states(t, df, raw if raw is not None else text)
    if states:
        out["state"] = states[0]
        out["state_a"] = states[0]
        if len(states) > 1:
            out["state_b"] = states[1]
    params = _params(t)
    if params:
        out["parameter"] = params[0]
    vars_ = list(params)
    if re.search(r"\boverall (?:score|scores)\b|\bscore\b", t) and "score" not in vars_ and len(vars_) < 2:
        vars_.append("score")
    if len(vars_) >= 2:
        out["x"], out["y"] = vars_[0], vars_[1]
    elif len(vars_) == 1:
        out["x"], out["y"] = vars_[0], "score"
    n = _top_n(t)
    if n:
        out["top_n"] = n
        out["top_k"] = min(n, 10)
    names, amb = _institutions(t, pool)
    if names:
        out["institution"] = names[0]
        if len(names) > 1:
            out["institution_b"] = names[1]
    elif amb:
        out["institution"] = amb  # engine will report the candidates
    if re.search(r"\b(?:bottom|worst|lowest ranked|lowest-ranked|least ranked)\b", t):
        out["bottom"] = True
    if re.search(r"\b(?:including|with|include|using) (?:band|bands|estimates?|estimated)\b", t):
        out["exact_only"] = False
    if re.search(r"\b(?:exact only|only exact|exact ranks only|without (?:band|bands|estimates))\b", t):
        out["exact_only"] = True
    if re.search(r"\b(?:no|without|hide) trend ?line\b", t):
        out["trendline"] = False
    return out
