"""Shared query engine + validator.

Every request (dashboard widget, chatbot, Gemini output) becomes a structured dict:
    {"question": "1.1", "category": "Engineering", "institution": "IIT Madras", "year_from": 2017, "year_to": 2025}
or, with only an "intent", the default question for that intent is used.
The engine validates every field against the data and the PARAM_SPECS allowlist, then calls a
registered analysis function. It NEVER executes generated code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go

from engine.data_loader import PARAMS, category_options, resolve_institution, state_options, year_range
from engine.registry import INTENTS, PARAM_SPECS, QUESTION_DEFAULTS, QUESTIONS

# make sure every analysis module has registered its questions
import analysis.geo, analysis.multivariate, analysis.stats, analysis.trends  # noqa: E401,F401

DEFAULT_QUESTION_FOR_INTENT = {
    "trend": "1.1", "compare": "2.4", "transition": "7.4", "state": "2.1",
    "map": "2.6", "distribution": "3.1", "correlation": "4.3", "gap": "5.5",
}
META_KEYS = {"intent", "question"}


class QueryValidationError(ValueError):
    """Raised with a user-friendly message when a structured query is invalid."""


class AmbiguousInstitution(QueryValidationError):
    def __init__(self, text, candidates):
        self.candidates = candidates
        super().__init__(f"'{text}' matches several institutions: " + "; ".join(candidates[:6]) + ". Please be more specific.")


@dataclass
class QueryResult:
    qid: str
    title: str
    params: dict
    data: pd.DataFrame
    figure: go.Figure
    notes: list[str] = field(default_factory=list)


def _int(v, name, lo, hi):
    try:
        iv = int(float(v))
    except (TypeError, ValueError):
        raise QueryValidationError(f"{name} must be a number.")
    if not lo <= iv <= hi:
        raise QueryValidationError(f"{name} must be between {lo} and {hi}.")
    return iv


def default_params(df: pd.DataFrame, qid: str) -> dict:
    y0, y1 = year_range(df)
    out = {}
    for p in QUESTIONS[qid].params:
        spec = PARAM_SPECS[p]
        d = spec.get("default")
        if p in ("year", "year_b", "year_to"):
            d = y1
        elif p in ("year_from",):
            d = y0
        elif p == "year_a":
            d = y1 - 1
        out[p] = QUESTION_DEFAULTS.get(qid, {}).get(p, d)
    return out


def validate_query(df: pd.DataFrame, query: dict) -> tuple[str, dict]:
    """Return (question_id, clean_params) or raise QueryValidationError."""
    if not isinstance(query, dict):
        raise QueryValidationError("The query must be a structured dictionary.")
    qid = query.get("question")
    intent = query.get("intent")
    if qid is None:
        if intent not in INTENTS:
            raise QueryValidationError(f"Unknown intent '{intent}'.")
        qid = DEFAULT_QUESTION_FOR_INTENT[intent]
    qid = str(qid)
    if qid not in QUESTIONS:
        raise QueryValidationError(f"Unknown question id '{qid}'.")
    q = QUESTIONS[qid]
    extra = set(query) - META_KEYS - set(PARAM_SPECS)
    if extra:
        raise QueryValidationError(f"Unsupported fields: {sorted(extra)}.")

    y0, y1 = year_range(df)
    params = default_params(df, qid)
    cats, states = category_options(df), state_options(df)
    category = query.get("category", params.get("category", "Overall"))
    if category not in cats:
        raise QueryValidationError(f"Unknown category '{category}'. Choose from: {', '.join(cats)}.")

    for key, val in query.items():
        if key in META_KEYS or key not in q.params or val is None:
            continue
        kind = PARAM_SPECS[key]["kind"]
        spec = PARAM_SPECS[key]
        if kind == "category":
            params[key] = category
        elif kind == "year":
            params[key] = _int(val, spec["label"], y0, y1)
        elif kind == "state":
            match = [s for s in states if s.lower() == str(val).lower()]
            if not match:
                raise QueryValidationError(f"Unknown state '{val}'.")
            params[key] = match[0]
        elif kind == "choice":
            ch = {str(c).lower(): c for c in spec["choices"]}
            if str(val).lower() not in ch:
                raise QueryValidationError(f"{spec['label']} must be one of {spec['choices']}.")
            params[key] = ch[str(val).lower()]
        elif kind == "int":
            params[key] = _int(val, spec["label"], spec["min"], spec["max"])
        elif kind == "float":
            try:
                fv = float(val)
            except (TypeError, ValueError):
                raise QueryValidationError(f"{spec['label']} must be a number.")
            params[key] = min(max(fv, spec["min"]), spec["max"])
        elif kind == "bool":
            params[key] = bool(val)
        elif kind == "institution":
            hit = resolve_institution(df, str(val), category)
            if hit.empty:
                raise QueryValidationError(f"I could not find '{val}' ranked in {category}.")
            if len(hit) > 1:
                raise AmbiguousInstitution(str(val), hit["institute_name"].tolist())
            params[key] = hit["institution_key"].iloc[0]
    if "category" in q.params:
        params["category"] = category
    if "institution" in q.params and not params.get("institution"):
        raise QueryValidationError("Which institution do you mean? Please name one.")
    if "year_from" in params and "year_to" in params and params["year_from"] > params["year_to"]:
        params["year_from"], params["year_to"] = params["year_to"], params["year_from"]
    if "year_a" in params and "year_b" in params and params["year_a"] > params["year_b"]:
        params["year_a"], params["year_b"] = params["year_b"], params["year_a"]
    return qid, params


def run_query(df: pd.DataFrame, query: dict) -> QueryResult:
    qid, params = validate_query(df, query)
    q = QUESTIONS[qid]
    data, fig = q.fn(df, **params)
    return QueryResult(qid=qid, title=q.title, params=params, data=data, figure=fig)


def run_question(df: pd.DataFrame, qid: str, **params) -> QueryResult:
    """Used by the dashboard pages: widgets already hold valid values, but we still validate."""
    return run_query(df, {"question": qid, **params})
