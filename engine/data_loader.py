"""
engine/data_loader.py  --  Shared NIRF data loader  (owner: M6)

Single entry point for ALL modules (M1 NLP, M3 trends, M4 geo, M5 stats, app).
Nobody should call pd.read_csv on the master file directly.

    from engine.data_loader import load_master, filter_data, rank_column

    df = load_master()                                   # cached, validated
    eng = filter_data(df, category="Engineering", years=(2020, 2025))
    col = rank_column(exact_only=True)                   # 'rank' or 'rank_value'

Rules baked in (agreed from the data review):
  * rank          -> exact NIRF rank only (NaN for banded / unpublished rows)
  * rank_value    -> exact rank, else band midpoint (an ESTIMATE)
  * rank_status   -> exact | band | not_published | withdrawn
  * Never compare scores across different `score_comparison_group` values.
  * Never aggregate across categories (an institution has ~1.5 rows per year).
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

# ----------------------------------------------------------------------------
# Constants shared by every module
# ----------------------------------------------------------------------------
DEFAULT_FILENAME = "nirf_master_final.csv"
ENV_VAR = "NIRF_DATA_PATH"

PARAMS = ["TLR", "RPC", "GO", "OI", "PR"]
PARAM_NAMES = {
    "TLR": "Teaching, Learning & Resources",
    "RPC": "Research & Professional Practice",
    "GO": "Graduation Outcomes",
    "OI": "Outreach & Inclusivity",
    "PR": "Perception",
}

REQUIRED_COLUMNS = [
    "institution_key", "institute_name", "year", "category", "category_group",
    "city", "state", "rank", "rank_low", "rank_high", "rank_midpoint",
    "rank_value", "rank_status", "is_rank_band", "score", *PARAMS,
    "score_comparison_group",
]
VALID_RANK_STATUS = {"exact", "band", "not_published", "withdrawn"}
BOOL_COLUMNS = [
    "is_rank_band", "is_top10", "is_top50", "is_top100", "is_new_entry",
    "has_score", "has_components",
]

DATA_NOTES = {
    "bands": "Band ranks (e.g. 101-150) are estimated by their midpoint and marked as estimates.",
    "scores": "Scores are only comparable within the same category and weighting scheme.",
    "coverage": "Exact ranks and scores are published only for the top part of each list.",
}


class DataValidationError(Exception):
    """Raised when the master file breaks a hard rule."""


# ----------------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------------
def resolve_path(path: str | os.PathLike | None = None) -> Path:
    """Explicit arg > $NIRF_DATA_PATH > common project locations."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    if os.environ.get(ENV_VAR):
        candidates.append(Path(os.environ[ENV_VAR]))
    here = Path(__file__).resolve().parent
    candidates += [
        Path("data") / DEFAULT_FILENAME,
        here.parent / "data" / DEFAULT_FILENAME,
        here / DEFAULT_FILENAME,
        Path(DEFAULT_FILENAME),
    ]
    for c in candidates:
        if c.exists():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find {DEFAULT_FILENAME}. Set {ENV_VAR} or put it in data/.\nTried:\n  {tried}"
    )


# ----------------------------------------------------------------------------
# Loading, cleaning, validating
# ----------------------------------------------------------------------------
def _read_master_uncached(path_str: str, mtime: float) -> pd.DataFrame:  # mtime busts cache on file change
    df = pd.read_csv(path_str)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Master file is missing columns: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64").astype(int)
    for c in ["rank", "rank_low", "rank_high", "rank_midpoint", "rank_value", "score", *PARAMS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BOOL_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1"])

    # --- derived helpers used across modules ---
    df["rank_is_estimate"] = df["rank_status"].eq("band")
    df["institution_label"] = df["institute_name"] + " (" + df["city"].fillna("") + ")"

    validate(df)
    return df.sort_values(["category", "year", "rank_value", "institute_name"]).reset_index(drop=True)


def _make_cached():
    """Use Streamlit's cache when running in the app, lru_cache elsewhere (notebooks, tests)."""
    try:
        import streamlit as st  # noqa: WPS433

        return st.cache_data(show_spinner=False)(_read_master_uncached)
    except Exception:  # streamlit not installed / not running
        return functools.lru_cache(maxsize=4)(_read_master_uncached)


_read_master = _make_cached()


def load_master(path: str | os.PathLike | None = None, *, include_bands: bool = True) -> pd.DataFrame:
    """
    Return the validated master dataset (a fresh copy, safe to mutate).

    include_bands=False keeps only rows with an exact published rank
    (use this for 'exact rank only' analysis).
    """
    p = resolve_path(path)
    df = _read_master(str(p), p.stat().st_mtime).copy()
    if not include_bands:
        df = df[df["rank_status"].eq("exact")].copy()
    return df


def validate(df: pd.DataFrame) -> list[str]:
    """Hard rules raise DataValidationError. Soft issues are returned as warnings."""
    errors, warnings = [], []

    if df.duplicated(["institution_key", "year", "category"]).any():
        errors.append("Duplicate institution/year/category rows.")
    bad_status = set(df["rank_status"].dropna()) - VALID_RANK_STATUS
    if bad_status:
        errors.append(f"Unknown rank_status values: {sorted(bad_status)}")
    if df["year"].lt(2017).any() or df["year"].gt(2100).any():
        errors.append("Year outside expected range.")

    bands = df[df["rank_status"].eq("band")]
    if bands[["rank_low", "rank_high", "rank_midpoint"]].isna().any().any():
        errors.append("Band rows without rank_low/rank_high/rank_midpoint.")
    exact = df[df["rank_status"].eq("exact")]
    if exact["rank"].isna().any():
        errors.append("Rows marked exact but missing rank.")
    for c in ["score", *PARAMS]:
        if ((df[c] < 0) | (df[c] > 100)).any():
            errors.append(f"{c} has values outside 0-100.")

    if df["institution_key"].isna().any() or df["state"].isna().any():
        warnings.append("Missing institution_key or state values.")

    if errors:
        raise DataValidationError(" | ".join(errors))
    return warnings


# ----------------------------------------------------------------------------
# Option lists for filters / dropdowns
# ----------------------------------------------------------------------------
def year_range(df: pd.DataFrame) -> tuple[int, int]:
    return int(df["year"].min()), int(df["year"].max())


def category_options(df: pd.DataFrame) -> list[str]:
    """General categories first (Overall, University, College ...), then disciplines."""
    g = df.drop_duplicates("category").set_index("category")["category_group"]
    order = {"General": 0, "Discipline": 1}
    return sorted(g.index, key=lambda c: (order.get(g[c], 2), c))


def state_options(df: pd.DataFrame) -> list[str]:
    return sorted(df["state"].dropna().unique())


def institution_options(df: pd.DataFrame, category: str | None = None, year: int | None = None) -> pd.DataFrame:
    """Unique institutions for a dropdown: columns institution_key, institute_name, state, city."""
    d = df
    if category:
        d = d[d["category"].eq(category)]
    if year:
        d = d[d["year"].eq(year)]
    cols = ["institution_key", "institute_name", "state", "city"]
    return d.drop_duplicates("institution_key")[cols].sort_values("institute_name").reset_index(drop=True)


ALIASES = {
    "iit": ("indian institute of technology",),
    "iits": ("indian institute of technology",),
    "nit": ("national institute of technology",),
    "nits": ("national institute of technology",),
    "iiit": ("indian institute of information technology", "international institute of information technology"),
    "iiits": ("indian institute of information technology", "international institute of information technology"),
    "iim": ("indian institute of management",),
    "iims": ("indian institute of management",),
    "iisc": ("indian institute of science",),
    "iiser": ("indian institute of science education and research",),
    "aiims": ("all india institute of medical sciences",),
    "bhu": ("banaras hindu university",),
    "jnu": ("jawaharlal nehru university",),
    "du": ("university of delhi",),
    "amu": ("aligarh muslim university",),
    "vit": ("vellore institute of technology",),
    "bits": ("birla institute of technology and science",),
    "srm": ("srm institute of science and technology", "s r m institute of science and technology"),
    "mahe": ("manipal academy of higher education",),
    "trichy": ("tiruchirappalli",),
    "trichi": ("tiruchirappalli",),
    "vizag": ("visakhapatnam",),
    "bangalore": ("bengaluru",),
    "calcutta": ("kolkata",),
    "madras": ("madras",),
}


def _norm(text: str) -> str:
    """Lower-case, '&' -> 'and', drop punctuation/hyphens, collapse spaces."""
    t = str(text).lower().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _expansions(text: str) -> list[str]:
    """All alias-expanded variants of a query (iiit -> indian / international ...)."""
    variants = [[]]
    for tok in _norm(text).split():
        opts = ALIASES.get(tok, (tok,))
        variants = [v + [o] for v in variants for o in opts]
    return [" ".join(v) for v in variants]


def resolve_institution(df: pd.DataFrame, text: str, category: str | None = None) -> pd.DataFrame:
    """
    Map free text (chatbot input) to institutions. Returns candidate rows
    (institution_key, institute_name, state, city); 1 row = unambiguous.
    Order: exact key > exact name > normalised name > alias-expanded equal >
    alias-expanded contains > raw contains (>=4 chars) > all words present.
    A bare acronym such as 'iit' therefore returns every IIT (ambiguous), not
    names that merely contain the letters 'iit' (e.g. IIITs).
    """
    pool = institution_options(df, category)
    t = str(text).strip()
    if not t:
        return pool.iloc[0:0]
    names = pool["institute_name"].map(_norm)
    raw = _norm(t)
    exps = _expansions(t)
    masks = [pool["institution_key"].eq(t), pool["institute_name"].eq(t), names.eq(raw)]
    masks += [names.eq(e) for e in exps]
    masks += [names.str.contains(r"\b" + re.escape(e) + r"\b", regex=True) for e in exps]
    if len(raw) >= 4:
        masks.append(names.str.contains(r"\b" + re.escape(raw) + r"\b", regex=True))
    for e in exps:
        words = e.split()
        masks.append(names.apply(lambda n, w=words: all(x in n.split() for x in w)))
    for mask in masks:
        hit = pool[mask]
        if not hit.empty:
            return hit.reset_index(drop=True)
    return pool.iloc[0:0]


# ----------------------------------------------------------------------------
# Filtering (the one place filter logic lives)
# ----------------------------------------------------------------------------
def rank_column(exact_only: bool = False) -> str:
    """'rank' = exact only; 'rank_value' = exact + band midpoint estimates."""
    return "rank" if exact_only else "rank_value"


def filter_data(
    df: pd.DataFrame,
    *,
    years: int | Sequence[int] | None = None,
    category: str | None = None,
    categories: Iterable[str] | None = None,
    states: str | Iterable[str] | None = None,
    institutions: str | Iterable[str] | None = None,
    exact_only: bool = False,
    require_score: bool = False,
) -> pd.DataFrame:
    """
    years        : int, (start, end) tuple = inclusive range, or list of years
    institutions : institution_key(s) or exact institute_name(s)
    exact_only   : keep only rows with an exact published rank
    require_score: drop rows without an overall score
    """
    d = df
    if years is not None:
        if isinstance(years, int):
            d = d[d["year"].eq(years)]
        elif isinstance(years, tuple) and len(years) == 2:
            d = d[d["year"].between(years[0], years[1])]
        else:
            d = d[d["year"].isin(list(years))]
    if category:
        d = d[d["category"].eq(category)]
    if categories:
        d = d[d["category"].isin(list(categories))]
    if states:
        s = [states] if isinstance(states, str) else list(states)
        d = d[d["state"].isin(s)]
    if institutions:
        i = [institutions] if isinstance(institutions, str) else list(institutions)
        d = d[d["institution_key"].isin(i) | d["institute_name"].isin(i)]
    if exact_only:
        d = d[d["rank_status"].eq("exact")]
    if require_score:
        d = d[d["score"].notna()]
    return d.copy()


# ----------------------------------------------------------------------------
# Guard-rails for analysis modules
# ----------------------------------------------------------------------------
def single_score_group(df: pd.DataFrame) -> str | None:
    """Return the one score_comparison_group in df, or None if mixed.
    Use before comparing scores across years (avoids the College-2017 weighting trap)."""
    groups = df["score_comparison_group"].dropna().unique()
    return groups[0] if len(groups) == 1 else None


def split_by_score_group(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One DataFrame per comparable scoring scheme."""
    return {g: d.copy() for g, d in df.dropna(subset=["score_comparison_group"]).groupby("score_comparison_group")}


def coverage_summary(df: pd.DataFrame) -> dict:
    """Small dict for a 'data coverage' caption in the UI."""
    return {
        "rows": len(df),
        "institutions": df["institution_key"].nunique(),
        "years": f"{df['year'].min()}-{df['year'].max()}" if len(df) else "-",
        "exact_rank_rows": int(df["rank_status"].eq("exact").sum()),
        "band_rows": int(df["rank_status"].eq("band").sum()),
        "score_rows": int(df["score"].notna().sum()),
    }


# ----------------------------------------------------------------------------
# Self-check:  python -m engine.data_loader   (or python data_loader.py)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_master()
    print("Loaded:", coverage_summary(data))
    print("Categories:", category_options(data))
    print("Years:", year_range(data), "| States:", len(state_options(data)))
    eng = filter_data(data, category="Engineering", years=2025, exact_only=True)
    print("Engineering 2025 exact rows:", len(eng))
    print(resolve_institution(data, "iit madras", "Overall")[["institution_key", "institute_name"]].to_string(index=False))
    print("Score groups in College:", list(split_by_score_group(filter_data(data, category="College"))))
    print("Warnings:", validate(data) or "none")
