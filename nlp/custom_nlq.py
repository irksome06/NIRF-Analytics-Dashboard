"""Ask NIRF -> chart-builder spec.

For questions that are not one of the 40 registered analyses ("average TLR of engineering colleges by state in 2022",
"number of institutions per city in the top 100", "lowest perception score institutions in pharmacy").
Pure rules, no model: the reader looks for a measure, an aggregation, a grouping, a chart type and filters,
and returns a spec for analysis.custom.build_chart (which validates it again against the data).

parse_custom(...) -> (spec | None, strength)   strength: "inherit" (follow-up of a previous custom chart),
                                                          "strong" (clearly a custom request: beats the registered match),
                                                          "weak"   (only used when nothing else matched)
"""
from __future__ import annotations

import re

import pandas as pd

from nlp.entities import CATEGORY_TERMS, _params, _states
from nlp.preprocess import normalise

GROUP_PATTERNS = [
    ("state", r"\b(?:by|per|each|every|across|for each|in each|of each) (?:the )?states?\b|\bstate ?wise\b|\bwhich states?\b|\bstates? (?:with|having|has|have)\b|\btop \d+ states\b"),
    ("category", r"\b(?:by|per|each|every|across|for each|in each) (?:the )?(?:categor(?:y|ies)|ranking categor(?:y|ies))\b|\bcategory ?wise\b|\bwhich categor(?:y|ies)\b"),
    ("city", r"\b(?:by|per|each|every|across|for each|in each) (?:the )?cit(?:y|ies)\b|\bcity ?wise\b|\bwhich cit(?:y|ies)\b|\bcit(?:y|ies) (?:with|having|has|have)\b|\btop \d+ cities\b"),
    ("year", r"\b(?:by|per|each|every|for each) (?:year|edition)\b|\byear ?wise\b|\bover (?:the )?(?:years|time|editions)\b|\btrend\b|\byearly\b|\bacross (?:the )?(?:years|editions)\b|\bevery year\b|\bsince 20\d\d\b|\bfrom 20\d\d to 20\d\d\b"),
    ("institution", r"\bwhich (?:institutions?|colleges?|universit(?:y|ies)|institutes?)\b|\b(?:institutions?|colleges?|universities|institutes?|schools?) (?:with|having|has|have|by|in terms of)\b|"
                    r"\b(?:top|best|lowest|highest|worst|bottom|weakest|strongest) \d* ?(?:institutions?|colleges?|universit(?:y|ies)|institutes?|schools?)\b|\bby institution\b|\bper institution\b"),
]
NAME_RX = re.compile(r"\b(iits?|nits?|iims?|iiits?|iisers?|aiims|nlus?|bits)\b")
HIGH = r"\b(?:highest|best|most|top|largest|strongest|greatest|maximum|max|leading|biggest)\b"
LOW = r"\b(?:lowest|worst|least|bottom|smallest|weakest|fewest|minimum|min|lower)\b"
COUNT = r"\b(?:number of|how many|count of|count|total number of|no of)\b"
INST_NOUN = r"institutions?|colleges?|universit(?:y|ies)|institutes?|schools?|ranked"
SCATTER_CUE = r"\b(?:vs|versus|against|scatter|relationship|correlat\w*|compare|comparison|relate\w*)\b"


def _all_categories(t: str) -> list[str]:
    seen: list[str] = []
    for term, cat in CATEGORY_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", t) and cat not in seen:
            seen.append(cat)
    if re.search(r"\buniversity\b", t) and "University" not in seen:
        seen.append("University")
    if re.search(r"\b(?:state public|open|skill) universit(?:y|ies)\b", t) and "University" in seen and not re.search(r"\buniversit(?:y|ies) (?:category|overall)\b", t):
        seen.remove("University")
    if "College" in seen and len(seen) > 1 and re.search(r"\b(?:engineering|management|pharmacy|pharma|medical|dental|law|architecture|agriculture|mba) (?:colleges?|institutions?|schools?)\b", t):
        seen.remove("College")
    return seen


def _metrics(t: str, raw: str = "") -> list[str]:
    t = re.sub(rf"\branked (?:{INST_NOUN})\b", " ", t)
    t2 = re.sub(r"\bresearch (?:institutions?|institutes?|category)\b", " ", t)       # a category, not the RPC parameter
    t2 = re.sub(r"\bmanagement (?:institutions?|institutes?|colleges?)\b", " ", t2)
    found = _params(t2)
    if "GO" not in found and re.search(r"\bGO\b", raw):      # "go" is an everyday word: only read it as a parameter when typed in capitals
        found.append("GO")
    if re.search(r"\b(?:overall |total |nirf )?scores?\b", t2) and "score" not in found:
        found.append("score")
    if re.search(r"\b(?:ranks?|ranking|rankings|position)\b", t2) and not found and not re.search(COUNT, t2):
        found.append("rank")
    return found


def _rank_cut(t: str, group: str | None) -> int | None:
    m = re.search(r"\b(?:in|within|inside|among|from|of) (?:the )?top (\d{1,3})\b", t)
    if m:
        return int(m.group(1))
    m = re.search(rf"\btop (\d{{1,3}}) (?:{INST_NOUN})\b", t)
    if m and group not in ("institution", None):
        return int(m.group(1))
    return None


def _top_n(t: str, rank_cut: int | None) -> int | None:
    m = re.search(r"\b(?:top|best|worst|bottom|first|leading|lowest|highest|least|most) (\d{1,2})\b", t) or re.search(r"\b(\d{1,2}) (?:best|top|worst|bottom|lowest|highest)\b", t)
    if m:
        n = int(m.group(1))
        if rank_cut is not None and n == rank_cut:
            return None
        return n if 1 <= n <= 50 else None
    return None


def parse_custom(text: str, df: pd.DataFrame, ents: dict, raw: str | None = None, last: dict | None = None) -> tuple[dict | None, str]:
    t = normalise(text)
    spec: dict = {}
    strength = "weak"
    metrics = _metrics(t, raw or text)
    count_q = bool(re.search(COUNT, t))
    group = next((g for g, rx in GROUP_PATTERNS if re.search(rx, t)), None)
    cats = _all_categories(t)
    states = _states(t, df, raw or text)
    if group is None and not states and re.search(r"\bstates?\b", t) and not re.search(r"\bstate public\b", t):
        group = "state"
    elif group is None and re.search(r"\bcit(?:y|ies)\b", t):
        group = "city"
    name = NAME_RX.search(t)
    if name and not name.group(1).endswith("s") and ents.get("institution"):     # "IIT Madras" is one institution, "IITs" is a family
        name = None

    # ---- chart type
    chart = None
    notes: list[str] = []
    if re.search(r"\bbox ?(?:and whisker )?(?:plot|chart)s?\b|\bboxplot\b|\bwhisker\b", t):
        chart = "box"
    elif re.search(r"\bhistogram\b|\bfrequency distribution\b", t):
        chart = "histogram"
    elif re.search(r"\bheat ?map\b", t):
        chart = "heatmap"
    elif re.search(r"\bscatter\b", t):
        chart = "scatter"
    elif re.search(r"\bline (?:chart|graph|plot)\b", t):
        chart = "line"
    elif re.search(r"\bbar (?:chart|graph|plot)\b|\bpie\b|\bdonut\b", t):
        chart = "bar"
        if re.search(r"\bpie\b|\bdonut\b", t):
            notes.append("Pie charts are not offered; showing a bar chart instead.")

    # ---- measure
    if count_q or (not metrics and re.search(rf"\b(?:{INST_NOUN})\b", t) and group):
        spec["metric"] = "count"
    elif metrics:
        spec["metric"] = metrics[0]
        if len(metrics) > 1 and (chart == "scatter" or re.search(SCATTER_CUE, t)) and not group:
            spec["metric_y"] = metrics[1]
    # ---- aggregation
    if re.search(r"\b(?:average|averages|avg|mean)\b", t):
        spec["agg"] = "mean"
    elif re.search(r"\bmedian\b", t):
        spec["agg"] = "median"
    elif re.search(r"\b(?:total|sum)\b", t) and spec.get("metric") != "count":
        spec["agg"] = "sum"
    elif re.search(r"\b(?:peak|ever|all time|at best)\b", t):
        spec["agg"] = "max" if spec.get("metric") != "rank" else "min"
    elif re.search(r"\b(?:at worst)\b", t):
        spec["agg"] = "min" if spec.get("metric") != "rank" else "max"

    # ---- grouping
    if group:
        spec["group_by"] = group
    if len(states) >= 2 and not group and spec.get("metric") != "count":
        group = spec["group_by"] = "state"
    if len(cats) >= 2 and group in (None, "category"):
        spec["group_by"], spec["categories"] = "category", cats
        group = "category"
    elif len(cats) >= 2 and group in ("year", "state", "city"):
        spec["split_by"], spec["categories"] = "category", cats
    elif cats and group != "category":
        spec["category"] = cats[0]
    elif cats and group == "category" and len(cats) == 1 and not re.search(r"\bfor\b", t):
        pass
    if len(states) >= 2 and group == "year":
        spec["split_by"], spec["states"] = "state", states
    elif states and not (group == "state" and not re.search(r"\b(?:in|for|of|within|from)\b", t)):
        spec["states"] = states
    elif states:
        spec["states"] = states
    if name:
        w = name.group(1).upper().rstrip("S") if name.group(1) not in ("aiims", "bits") else name.group(1).upper()
        spec["name_contains"] = w
    elif ents.get("institution") and isinstance(ents["institution"], str):
        spec["institutions"] = [ents["institution"]]

    # ---- years
    yf, yt = ents.get("year_from"), ents.get("year_to")
    if yf and yt and yf != yt and re.search(r"\b(?:from|between|since|last|past|recent|to)\b|20\d\d ?-", t):
        spec["year_from"], spec["year_to"] = yf, yt
    elif ents.get("year"):
        spec["year"] = ents["year"]
    elif yf and yt:
        spec["year_from"], spec["year_to"] = yf, yt
    if "year" in spec and group == "year":      # "trend ... in 2022" makes no sense as one point: keep as a filter-free trend
        spec.pop("year")

    # ---- ranking cut, top n, direction
    g_now = spec.get("group_by")
    rc = _rank_cut(t, g_now)
    if rc:
        spec["rank_max"] = rc
    tn = _top_n(t, rc)
    if tn:
        spec["top_n"] = tn
    elif ents.get("top_n") and not rc:
        spec["top_n"] = ents["top_n"]
    want_high, want_low = re.search(HIGH, t), re.search(LOW, t)
    if want_high or want_low:
        if spec.get("metric") == "rank":
            spec["sort"] = "asc" if (want_high and not want_low) else "desc"
        else:
            spec["sort"] = "desc" if (want_high and not want_low) else "asc"
    if ents.get("exact_only") is False:
        spec["exact_only"] = False
    if not chart and not group and re.search(r"\b(?:distribution|distributed|spread|histogram)\b", t) and spec.get("metric") in ("TLR", "RPC", "GO", "OI", "PR", "score", "rank"):
        chart = "histogram"
    if chart:
        spec["chart"] = chart

    # ---- follow-up of a previous custom chart: inherit what the new message does not state
    fresh = {k for k in spec if k not in ("category", "year", "year_from", "year_to", "states", "top_n", "exact_only", "sort")}
    if last and not fresh and len(t.split()) <= 8:
        merged = {k: v for k, v in last.items() if k != "question"}
        for k in ("category", "categories", "year", "year_from", "year_to", "states", "top_n", "exact_only", "sort"):
            if k in spec:
                if k == "category":
                    merged.pop("categories", None)
                if k in ("year",):
                    merged.pop("year_from", None), merged.pop("year_to", None)
                if k in ("year_from", "year_to"):
                    merged.pop("year", None)
                merged[k] = spec[k]
        if any(k in spec for k in ("category", "year", "year_from", "year_to", "states", "top_n", "sort", "exact_only")):
            return merged, "inherit"
    if last and re.match(r"^(?:and |also |now |instead |but |same |show it |make it |as an? )", t) and spec:
        merged = {k: v for k, v in last.items() if k != "question"}
        merged.update(spec)
        if spec.get("group_by") == "year":
            for k in ("year", "year_from", "year_to"):
                if k not in spec:
                    merged.pop(k, None)
        return merged, "inherit"

    # ---- institution rankings by a measure: "lowest perception score institutions", "top 5 universities by research score"
    ranking_noun = re.search(rf"\b(?:{INST_NOUN})\b", t) and (want_high or want_low) and "group_by" not in spec
    if ranking_noun and spec.get("metric") not in (None, "count"):
        spec["group_by"] = "institution"
    if spec.get("metric") is None and not group and not chart:
        spec = {}
    if "metric" not in spec and "group_by" in spec:
        return None, "weak"

    if not spec:
        return None, "weak"
    if chart == "scatter" and "metric_y" not in spec:
        return None, "weak"

    # ---- how sure are we that this is a custom question rather than one of the 40?
    # Strong = an explicit "<measure> by/per <group>" style request the registered analyses do not cover.
    registered_topic = re.search(r"\b(?:new\w*|appear\w*|entr\w+|moved|move\w*|stayed|stay\w*|retain\w*|improv\w*|declin\w*|climb\w*|fell|fall\w*|drop\w*|volatil\w*|outlier\w*|unusual\w*|correlat\w*|gap|gaps|jump\w*|mover\w*|"
                                r"promot\w*|consisten\w*|always|despite|catching|exit\w*|stable|unstable|quartile\w*|bands?|relative|rest|versus|vs)\b", t)
    both_dirs = bool(want_high and want_low) or bool(re.search(r"\bhigh\b.*\blow\b|\blow\b.*\bhigh\b", t))
    explicit = bool(re.search(r"\b(?:by|per|each|every|wise)\b", t)) or bool(re.search(r"\b(?:average|mean|median)\b", t))
    param_metric = spec.get("metric") in ("TLR", "RPC", "GO", "OI", "PR")
    g = spec.get("group_by")
    if not registered_topic and not both_dirs:
        if (param_metric and g in ("state", "category", "city") and explicit) \
                or (param_metric and g == "year" and re.search(r"\b(?:average|mean|median)\b", t)) \
                or (g == "institution" and param_metric and (want_high or want_low) and not spec.get("metric_y")) \
                or g == "city" \
                or spec.get("chart") in ("box", "heatmap") \
                or (spec.get("chart") == "histogram" and param_metric) \
                or ("name_contains" in spec) or ("split_by" in spec) \
                or (spec.get("metric") == "count" and g in ("state", "category", "city", "year") and "rank_max" not in spec and not re.search(r"\b(?:top|best|leading)\b", t) and "institutions" not in spec):
            strength = "strong"
    if notes:
        spec["_notes"] = notes
    return spec, strength
