"""Institution explorer helpers (profile radar and multi-institution comparison). Same contract: fn(df, **filters) -> (table, figure)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from engine.data_loader import PARAM_NAMES, PARAMS
from engine.helpers import C, SERIES, empty_fig, rank_col, scope, style


def institution_radar(df, category="Overall", institution=None, year=2025, **_):
    """Parameter profile of one institution against the cohort median and the top-10 average."""
    d = scope(df, category, year, exact_only=True, require_score=True)
    me = d[(d["institution_key"] == institution) | (d["institute_name"] == institution)]
    if me.empty or len(d) < 5:
        return pd.DataFrame(), empty_fig("No scored data for this institution in the selected year.")
    me = me.iloc[0]
    top = d.nsmallest(10, "rank")
    prof = pd.DataFrame({"parameter": PARAMS, "institution": [me[p] for p in PARAMS],
                         "cohort_median": [d[p].median() for p in PARAMS], "top10_average": [top[p].mean() for p in PARAMS]}).round(1)
    fig = go.Figure()
    for col, name, color, dash in [("top10_average", "Top-10 average", C["slate"], "dot"), ("cohort_median", "Cohort median", C["amber"], "dash"),
                                   ("institution", me["institute_name"], C["brand"], "solid")]:
        fig.add_trace(go.Scatterpolar(r=list(prof[col]) + [prof[col].iloc[0]], theta=PARAMS + [PARAMS[0]], name=name,
                                      line=dict(color=color, dash=dash, width=3 if col == "institution" else 2),
                                      fill="toself" if col == "institution" else None, fillcolor="rgba(15,123,138,.18)"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor="#E3E4EC")))
    style(fig, f"{me['institute_name']}: parameter profile {year}", f"Rank {int(me['rank'])} · score {me['score']:.1f} · {category}", height=480)
    return prof, fig


def compare_institutions(df, category="Overall", institutions=(), year_from=2017, year_to=2025, exact_only=True, **_):
    """Rank paths of several institutions on one chart (band estimates drawn as hollow diamonds)."""
    col = rank_col(exact_only)
    d = scope(df, category, year_from=year_from, year_to=year_to, exact_only=exact_only)
    d = d[(d["institution_key"].isin(list(institutions)) | d["institute_name"].isin(list(institutions))) & d[col].notna()].sort_values("year")
    if d.empty:
        return pd.DataFrame(), empty_fig("Pick at least one institution that is ranked in this category.")
    fig = go.Figure()
    for i, (_, g) in enumerate(d.groupby("institution_key", sort=False)):
        color, name = SERIES[i % len(SERIES)], g["institute_name"].iloc[0]
        fig.add_trace(go.Scatter(x=g["year"], y=g[col], mode="lines+markers", name=name, line=dict(color=color, width=2.5),
                                 marker=dict(size=9, color=color, symbol=["circle" if s == "exact" else "diamond-open" for s in g["rank_status"]], line=dict(width=2, color=color)),
                                 hovertemplate=name + "<br>%{x}: rank %{y:.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="NIRF rank (1 = best)")
    fig.update_xaxes(dtick=1)
    style(fig, f"Rank comparison, {category}", "Hollow diamonds are band midpoint estimates" if not exact_only else "Exact published ranks only", height=460)
    return d[["year", "institute_name", "rank", "rank_value", "rank_status", "score"]].reset_index(drop=True), fig
