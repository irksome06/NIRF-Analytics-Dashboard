"""M4 - Geography: Section 2 (state analysis) and the India map."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from engine.helpers import C, empty_fig, scope, style
from engine.registry import question

GEOJSON_PATH = Path(__file__).resolve().parent.parent / "data" / "india_states.geojson"   # simplified state boundaries (146 KB, offline)

# approximate state / UT centroids (lat, lon) for the bubble map - no external GeoJSON needed
CENTROIDS = {
    "Andhra Pradesh": (15.9, 79.7), "Arunachal Pradesh": (28.2, 94.7), "Assam": (26.2, 92.9), "Bihar": (25.1, 85.3),
    "Chhattisgarh": (21.3, 81.9), "Goa": (15.3, 74.1), "Gujarat": (22.3, 71.2), "Haryana": (29.1, 76.1),
    "Himachal Pradesh": (31.9, 77.2), "Jharkhand": (23.6, 85.3), "Karnataka": (15.3, 75.7), "Kerala": (10.5, 76.3),
    "Madhya Pradesh": (23.5, 78.7), "Maharashtra": (19.7, 75.7), "Manipur": (24.7, 93.9), "Meghalaya": (25.5, 91.4),
    "Mizoram": (23.2, 92.8), "Nagaland": (26.1, 94.5), "Odisha": (20.5, 84.4), "Punjab": (31.0, 75.3),
    "Rajasthan": (26.6, 73.8), "Sikkim": (27.6, 88.5), "Tamil Nadu": (11.1, 78.4), "Telangana": (17.9, 79.0),
    "Tripura": (23.7, 91.7), "Uttar Pradesh": (26.8, 80.9), "Uttarakhand": (30.1, 79.2), "West Bengal": (23.0, 87.9),
    "Delhi": (28.6, 77.2), "Chandigarh": (30.73, 76.78), "Puducherry": (11.9, 79.8), "Jammu and Kashmir": (33.8, 76.6),
    "Andaman and Nicobar Islands": (11.7, 92.7), "Dadra and Nagar Haveli and Daman and Diu": (20.4, 72.9),
}


def _top100(d):
    return d[d["is_top100"]]


def _counts(df, category, year, top100=True):
    d = scope(df, category, year)
    d = _top100(d) if top100 else d[d["rank_status"].isin(["exact", "band"])]
    return d.groupby("state").size().rename("institutions").reset_index().sort_values("institutions", ascending=False)


@question("2.1", "Which states have the highest number of institutions in the top 100?", "state",
          ["category", "year", "top_n"],
          ["highest number of institutions", "states have the most", "top 100 by state", "which states", "most institutions in the top 100", "state with most"])
def top_states(df, category="Overall", year=2025, top_n=10, **_):
    allc = _counts(df, category, year)
    total = int(allc["institutions"].sum())              # national total, not just the states shown
    c = allc.head(top_n).assign(share_pct=lambda t: (100 * t["institutions"] / total).round(1)) if total else allc.head(top_n)
    if c.empty:
        return c, empty_fig("No top-100 institutions for this category and year.")
    c = c.sort_values("institutions")
    fig = go.Figure(go.Bar(y=c["state"], x=c["institutions"], orientation="h", marker_color=C["brand"],
                           text=c["institutions"], textposition="outside"))
    fig.update_xaxes(title="Institutions in the top 100")
    style(fig, f"States with the most top-100 institutions, {category} {year}", height=max(360, 32 * len(c) + 140))
    return c.sort_values("institutions", ascending=False).reset_index(drop=True), fig


@question("2.2", "How is the geographical distribution of top-ranked institutions changing from 2017 to 2025?", "state",
          ["category", "year_from", "year_to", "top_n"],
          ["geographical distribution", "distribution of top-ranked", "changing geographically", "state distribution over time", "regional distribution"])
def distribution_over_time(df, category="Overall", year_from=2017, year_to=2025, top_n=15, **_):
    d = _top100(scope(df, category, year_from=year_from, year_to=year_to))
    if d.empty:
        return pd.DataFrame(), empty_fig("No top-100 institutions in this period.")
    t = d.groupby(["state", "year"]).size().rename("institutions").reset_index()
    pv = t.pivot(index="state", columns="year", values="institutions").fillna(0).astype(int)
    pv = pv.loc[pv.iloc[:, -1].sort_values(ascending=False).index].head(top_n)
    pv = pv.loc[pv.sum(axis=1).sort_values(ascending=False).index]
    fig = go.Figure(go.Heatmap(z=pv.values, x=[str(c) for c in pv.columns], y=pv.index,
                               text=pv.replace(0, pd.NA).map(lambda v: "" if pd.isna(v) else str(int(v))).values,
                               texttemplate="%{text}", colorscale=[[0, "#F4F5FB"], [1, C["brand"]]], xgap=2, ygap=2,
                               showscale=False, hovertemplate="%{y}, %{x}: %{z} institutions<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    style(fig, f"Top-100 institutions by state over time, {category}", f"Top {len(pv)} states by total; blank cell = none.",
          height=max(380, 28 * len(pv) + 150))
    out = pv.copy()
    out.columns = [str(c) for c in out.columns]
    return out.reset_index(), fig


@question("2.3", "Which states have improved their representation in the top 100 over time?", "state",
          ["category", "year_a", "year_b"],
          ["improved their representation", "gained representation", "states improved", "representation in the top 100", "states gained"])
def state_change(df, category="Overall", year_a=2017, year_b=2025, **_):
    a = _counts(df, category, year_a).set_index("state")["institutions"]
    b = _counts(df, category, year_b).set_index("state")["institutions"]
    t = pd.concat([a.rename("earlier"), b.rename("later")], axis=1).fillna(0).astype(int)
    t["change"] = t["later"] - t["earlier"]
    t = t[t["change"] != 0].sort_values("change")
    if t.empty:
        return t.reset_index(), empty_fig("No state changed its top-100 count between these years.")
    fig = go.Figure(go.Bar(y=t.index, x=t["change"], orientation="h",
                           marker_color=[C["brand"] if v > 0 else C["coral"] for v in t["change"]],
                           customdata=t[["earlier", "later"]], text=t["change"].map("{:+d}".format), textposition="outside",
                           hovertemplate="%{y}: %{customdata[0]} → %{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title=f"Change in top-100 institutions, {year_a} → {year_b}")
    style(fig, f"Change in top-100 representation by state, {category}", "Teal = gained, coral = lost.",
          height=max(380, 28 * len(t) + 150))
    return t.sort_values("change", ascending=False).reset_index(), fig


@question("2.4", "How do the rankings of institutions in Karnataka compare with those in Tamil Nadu?", "compare",
          ["category", "state_a", "state_b", "year_from", "year_to", "exact_only"],
          ["compare karnataka", "karnataka with tamil nadu", "compare states", "state versus state", "karnataka vs tamil nadu", "compare the rankings of"])
def compare_states(df, category="Overall", state_a="Karnataka", state_b="Tamil Nadu", year_from=2017, year_to=2025, exact_only=True, **_):
    from plotly.subplots import make_subplots
    d = scope(df, category, year_from=year_from, year_to=year_to, states=[state_a, state_b])
    if d.empty:
        return pd.DataFrame(), empty_fig("Neither state has ranked institutions in this category.")
    col = "rank" if exact_only else "rank_value"
    top = _top100(d).groupby(["state", "year"]).size().rename("top100_count")
    med = d.dropna(subset=[col]).groupby(["state", "year"])[col].median().rename("median_rank")
    n = d.groupby(["state", "year"]).size().rename("ranked_institutions")
    t = pd.concat([top, med, n], axis=1).reset_index().fillna({"top100_count": 0})
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Institutions in the top 100", "Median rank of listed institutions (lower is better)"))
    for s, colr in ((state_a, C["brand"]), (state_b, C["amber"])):
        x = t[t["state"] == s]
        fig.add_trace(go.Scatter(x=x["year"], y=x["top100_count"], mode="lines+markers", name=s, line=dict(color=colr, width=2.5),
                                 legendgroup=s), row=1, col=1)
        fig.add_trace(go.Scatter(x=x["year"], y=x["median_rank"], mode="lines+markers", name=s, line=dict(color=colr, width=2.5),
                                 legendgroup=s, showlegend=False), row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(dtick=1)
    style(fig, f"{state_a} vs {state_b}, {category}", "Median rank uses " + ("exact ranks only." if exact_only else "exact ranks and band midpoints (estimates)."))
    for a in fig.layout.annotations:
        a.font = dict(size=12, color=C["slate"])
    return t.round(1), fig


@question("2.5", "Which states have produced the largest number of newly appearing institutions in the rankings?", "state",
          ["category", "year_from", "year_to", "top_n"],
          ["newly appearing", "new institutions", "new entrants", "newly ranked", "first appeared", "new entries"])
def new_entries(df, category="Overall", year_from=2017, year_to=2025, top_n=12, **_):
    d = scope(df, category, year_from=year_from, year_to=year_to)
    d = d[d["is_new_entry"]]
    if d.empty:
        return pd.DataFrame(), empty_fig("No newly appearing institutions in this period.")
    tot = d.groupby("state").size().sort_values(ascending=False)
    top = tot.head(top_n).index
    t = d[d["state"].isin(top)].groupby(["state", "year"]).size().rename("new_institutions").reset_index()
    fig = go.Figure()
    order = list(top)[::-1]
    from engine.helpers import SERIES
    for i, y in enumerate(sorted(t["year"].unique())):
        s = t[t["year"] == y].set_index("state").reindex(order)["new_institutions"].fillna(0)
        fig.add_trace(go.Bar(y=order, x=s.values, orientation="h", name=str(y), marker_color=(SERIES + ["#1A1D3A"] * 4)[i % 12]))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="New institutions (first appearance in this category)")
    style(fig, f"States producing the most new institutions, {category}",
          "A new institution is one appearing in a category list for the first time after that list's first year.",
          height=max(400, 30 * len(order) + 170))
    res = tot.rename("new_institutions").reset_index().head(top_n)
    return res, fig


@lru_cache(maxsize=1)
def _boundaries():
    with open(GEOJSON_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _boundary_map(df, category, year, map_metric, top100):
    """Real state-boundary choropleth; hover shows the count and the best-ranked institution of each state."""
    gj = _boundaries()
    props = [f["properties"] for f in gj["features"]]
    t = pd.DataFrame({"state": [p["state"] for p in props], "cx": [p["cx"] for p in props], "cy": [p["cy"] for p in props]})
    counts = _counts(df, category, year, top100=top100).set_index("state")["institutions"]
    t["institutions"] = t["state"].map(counts).fillna(0).astype(int)
    d = scope(df, category, year, exact_only=True).sort_values("rank")
    t["best"] = t["state"].map(d.groupby("state")["institute_name"].first()).fillna("-")
    if t["institutions"].sum() == 0:
        return t, empty_fig("No institutions to plot.")
    zmax = max(int(t["institutions"].max()), 1)
    fig = go.Figure(go.Choropleth(
        geojson=gj, locations=t["state"], featureidkey="properties.state", z=t["institutions"], zmin=0, zmax=zmax,
        colorscale=[[0, "#EFF0F7"], [0.0001, "#C9CEF0"], [0.4, "#6B76C9"], [1, C["ink"]]],
        marker_line_color="white", marker_line_width=0.8, customdata=t[["best"]],
        hovertemplate="<b>%{location}</b><br>%{z} institutions<br>Best ranked: %{customdata[0]}<extra></extra>",
        colorbar=dict(title="Institutions", thickness=12, len=0.6)))
    lab = t[t["institutions"] >= 2]
    fig.add_trace(go.Scattergeo(lon=lab["cx"], lat=lab["cy"], mode="text", text=lab["institutions"].astype(str), showlegend=False, hoverinfo="skip",
                                textfont=dict(size=11, family="Arial Black", color=["white" if v > 0.5 * zmax else C["ink"] for v in lab["institutions"]])))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    style(fig, f"{map_metric}, {category} {year}", "Hover a state for its count and best-ranked institution; numbers show states with 2 or more.", height=600)
    res = t[t["institutions"] > 0].sort_values("institutions", ascending=False)[["state", "institutions", "best"]].reset_index(drop=True)
    return res, fig


@question("2.6", "Show institutions across India on a map.", "map",
          ["category", "year", "map_metric", "map_style"],
          ["map", "india map", "on a map", "geographic map", "show the map", "map of india"])
def india_map(df, category="Overall", year=2025, map_metric="Institutions in top 100", map_style="Boundary map (offline)", **_):
    top100 = map_metric.startswith("Institutions in top")
    if str(map_style).startswith("Boundary") and GEOJSON_PATH.exists():
        return _boundary_map(df, category, year, map_metric, top100)
    c = _counts(df, category, year, top100=top100)
    c = c[c["state"].isin(CENTROIDS)].copy()
    if c.empty:
        return c, empty_fig("No institutions to plot.")
    c["lat"] = c["state"].map(lambda s: CENTROIDS[s][0])
    c["lon"] = c["state"].map(lambda s: CENTROIDS[s][1])
    size = 10 + 38 * (c["institutions"] / c["institutions"].max()) ** 0.5
    hover = "<b>%{customdata[0]}</b><br>%{customdata[1]} institutions<extra></extra>"
    marker = dict(size=size, color=c["institutions"], colorscale=[[0, "#C9CEF0"], [1, C["ink"]]], opacity=0.85,
                  line=dict(color="white", width=1))
    if map_style.startswith("Basemap"):
        fig = go.Figure(go.Scattergeo(lat=c["lat"], lon=c["lon"], mode="markers", marker=marker,
                                      customdata=c[["state", "institutions"]], hovertemplate=hover))
        fig.update_geos(lataxis_range=[6, 37.5], lonaxis_range=[67, 98.5], showcountries=True, countrycolor="#B4B8CC",
                        showland=True, landcolor="#EFF0F7", showocean=True, oceancolor="#E6E8F4", showframe=False,
                        projection_type="mercator", resolution=50)
        fig.update_layout(geo=dict(bgcolor="rgba(0,0,0,0)"))
    else:
        fig = go.Figure(go.Scatter(x=c["lon"], y=c["lat"], mode="markers+text", marker=marker,
                                   text=c["institutions"], textfont=dict(color="white", size=10),
                                   customdata=c[["state", "institutions"]], hovertemplate=hover))
        fig.update_yaxes(scaleanchor="x", scaleratio=1, visible=False)
        fig.update_xaxes(visible=False)
    style(fig, f"{map_metric}, {category} {year}", "Bubble area is proportional to the count; position is the state centroid.", height=600)
    return c[["state", "institutions"]].reset_index(drop=True), fig
