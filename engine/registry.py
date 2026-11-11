"""Question registry: every analysis function registers itself here once.

Contract for every registered function:
    fn(df, **filters) -> (result_df, plotly_figure)
The Streamlit pages, the query validator and the chatbot all read this registry,
so adding a question in one place makes it available everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from engine.data_loader import PARAMS

NUMERIC_VARS = ["score", "TLR", "RPC", "GO", "OI", "PR", "rank_value"]

# Widget / validation spec for every parameter a question may use.
# kind: category | year | institution | state | choice | int | float | bool
PARAM_SPECS: dict[str, dict] = {
    "category": dict(kind="category", label="Category", default="Overall"),
    "year": dict(kind="year", label="Year", default=2025),
    "year_from": dict(kind="year", label="From year", default=2017),
    "year_to": dict(kind="year", label="To year", default=2025),
    "year_a": dict(kind="year", label="Earlier year", default=2024),
    "year_b": dict(kind="year", label="Later year", default=2025),
    "institution": dict(kind="institution", label="Institution", default=None),
    "state": dict(kind="state", label="State", default="Tamil Nadu"),
    "state_filter": dict(kind="state", label="Only in state", default=None, optional=True),
    "institution_b": dict(kind="institution", label="Compare with", default=None, optional=True),
    "bottom": dict(kind="bool", label="Show the lowest-ranked instead", default=False),
    "state_a": dict(kind="state", label="State A", default="Karnataka"),
    "state_b": dict(kind="state", label="State B", default="Tamil Nadu"),
    "parameter": dict(kind="choice", label="Parameter", choices=PARAMS, default="RPC"),
    "x": dict(kind="choice", label="X variable", choices=NUMERIC_VARS, default="RPC"),
    "y": dict(kind="choice", label="Y variable", choices=NUMERIC_VARS, default="score"),
    "top_n": dict(kind="int", label="How many to show", min=3, max=30, default=10),
    "top_k": dict(kind="int", label="Institutions to compare", min=3, max=10, default=5),
    "min_obs": dict(kind="int", label="Minimum years ranked", min=2, max=9, default=4),
    "poor_q": dict(kind="float", label="Poor rank = worse than percentile", min=0.3, max=0.9, step=0.05, default=0.5),
    "min_years": dict(kind="int", label="Minimum years in top 10", min=2, max=9, default=3),
    "min_n": dict(kind="int", label="Minimum institutions per state", min=2, max=20, default=5),
    "exact_only": dict(kind="bool", label="Exact published ranks only", default=True),
    "trendline": dict(kind="bool", label="Show trend line", default=True),
    "standardize": dict(kind="bool", label="Compare on standardised (z) scores", default=False),
    "z": dict(kind="float", label="Outlier threshold (z)", min=1.0, max=4.0, step=0.25, default=2.0),
    "hi_q": dict(kind="float", label="High = top percentile", min=0.5, max=0.95, step=0.05, default=0.75),
    "lo_q": dict(kind="float", label="Low = below percentile", min=0.05, max=0.75, step=0.05, default=0.5),
    "min_decline": dict(kind="float", label="Material parameter decline (points)", min=0.0, max=10.0, step=0.5, default=2.0),
    "map_metric": dict(kind="choice", label="Map shows", choices=["Institutions in top 100", "All ranked institutions"], default="Institutions in top 100"),
    "map_style": dict(kind="choice", label="Map style", choices=["Boundary map (offline)", "Basemap (needs internet)", "Offline"], default="Boundary map (offline)"),
}
# Per-question default overrides (applied on top of PARAM_SPECS defaults)
QUESTION_DEFAULTS: dict[str, dict] = {
    "7.4": {"top_n": 8}, "5.4": {"top_n": 15}, "2.5": {"top_n": 12}, "2.2": {"top_n": 15},
    "8.3": {"z": 3.5}, "2.3": {"year_a": 2017, "year_b": 2025}, "7.1": {"year_a": 2024, "year_b": 2025}, "8.4": {"hi_q": 0.7, "poor_q": 0.4}, "8.1": {"z": 1.75},
}
# Parameters supplied by the sidebar when present (everything else is shown inline).
GLOBAL_PARAMS = ["category", "year", "year_from", "year_to", "exact_only"]
# Parameters that accept a free extra argument but are never required.
INTENTS = ["trend", "compare", "transition", "state", "map", "distribution", "correlation", "gap"]


@dataclass
class Question:
    qid: str
    section: int
    title: str
    intent: str
    fn: Callable
    params: list[str]
    keywords: list[str] = field(default_factory=list)
    help: str = ""


QUESTIONS: dict[str, Question] = {}

SECTION_TITLES = {
    1: "Time-series and trend analysis",
    2: "Geospatial and regional analysis",
    3: "Univariate analysis",
    4: "Bivariate analysis and correlation",
    5: "Multivariate analysis",
    6: "Distribution and statistical comparison",
    7: "Ranking transitions and institutional mobility",
    8: "Outlier detection and performance gaps",
}


def question(qid: str, title: str, intent: str, params: list[str], keywords: list[str], help: str = ""):
    """Decorator that registers an analysis function."""
    def wrap(fn):
        QUESTIONS[qid] = Question(qid, int(qid.split(".")[0]), title, intent, fn, params, keywords, help)
        return fn
    return wrap


def by_section(section: int) -> list[Question]:
    qs = [q for q in QUESTIONS.values() if q.section == section]
    return sorted(qs, key=lambda q: [int(p) for p in q.qid.split(".")])
