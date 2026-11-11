"""M5 - Statistics: Section 3 (univariate), Section 4 (bivariate/correlation), Section 6 (distribution comparison)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.data_loader import PARAM_NAMES, PARAMS
from engine.helpers import C, PARAM_COLORS, SERIES, empty_fig, iqr, ols, scope, style
from engine.registry import question

NAMES = {**PARAM_NAMES, "score": "Overall score", "rank_value": "Rank (band midpoints estimated)", "rank": "Rank"}


def _scores(df, category, year):
    return scope(df, category, year, require_score=True)


# ============================================================ SECTION 3
@question("3.1", "What is the distribution of NIRF scores for a selected year?", "distribution",
          ["category", "year"],
          ["distribution of scores", "score distribution", "histogram of scores", "distribution of nirf scores", "how are scores distributed"])
def score_distribution(df, category="Overall", year=2025, **_):
    d = _scores(df, category, year)
    if d.empty:
        return pd.DataFrame(), empty_fig("No scores are published for this category and year.")
    s = d["score"]
    fig = go.Figure(go.Histogram(x=s, nbinsx=15, marker_color=C["brand"], marker_line=dict(color="white", width=1),
                                 hovertemplate="score %{x}<br>%{y} institutions<extra></extra>"))
    fig.add_vline(x=s.mean(), line_color=C["amber"], line_width=2, annotation_text=f"mean {s.mean():.1f}", annotation_position="top")
    fig.add_vline(x=s.median(), line_color=C["coral"], line_dash="dash", annotation_text=f"median {s.median():.1f}", annotation_position="bottom")
    fig.update_xaxes(title="Overall score")
    fig.update_yaxes(title="Institutions")
    style(fig, f"Distribution of overall scores, {category} {year}", f"{len(s)} institutions with a published score.")
    return d[["institute_name", "state", "rank", "score"]].sort_values("score", ascending=False).reset_index(drop=True), fig


def _rank_bucket(row):
    if row["rank_status"] == "band":
        return str(row["rank_band_label"]), row["rank_low"], "Published band"
    r = row["rank"]
    for hi, lab in ((10, "1–10"), (25, "11–25"), (50, "26–50"), (75, "51–75"), (100, "76–100"), (150, "101–150"), (200, "151–200")):
        if r <= hi:
            return lab, r, "Exact rank"
    return "201+", r, "Exact rank"


@question("3.2", "What is the distribution of overall ranks across all institutions?", "distribution",
          ["category", "year"],
          ["distribution of ranks", "rank distribution", "distribution of overall ranks", "how many institutions in each rank"])
def rank_distribution(df, category="Overall", year=2025, **_):
    d = scope(df, category, year)
    d = d[d["rank_status"].isin(["exact", "band"])]
    if d.empty:
        return pd.DataFrame(), empty_fig("No ranked institutions for this category and year.")
    b = d.apply(_rank_bucket, axis=1, result_type="expand")
    b.columns = ["bucket", "low", "type"]
    t = b.groupby(["bucket", "type"]).agg(institutions=("low", "size"), low=("low", "min")).reset_index().sort_values("low")
    fig = go.Figure()
    for typ, col in (("Exact rank", C["brand"]), ("Published band", C["amber"])):
        s = t[t["type"] == typ]
        if not s.empty:
            fig.add_trace(go.Bar(x=s["bucket"], y=s["institutions"], name=typ, marker_color=col, text=s["institutions"], textposition="outside"))
    fig.update_xaxes(categoryorder="array", categoryarray=t["bucket"].tolist(), title="Rank group")
    fig.update_yaxes(title="Institutions")
    style(fig, f"Institutions by rank group, {category} {year}",
          "Exact ranks are grouped for readability; beyond the published top list NIRF only reports bands.")
    return t.drop(columns="low"), fig


@question("3.3", "What is the average, median, minimum, and maximum score in a given year?", "distribution",
          ["category", "year"],
          ["average median minimum maximum", "mean median min max", "summary statistics", "average, median", "descriptive statistics", "minimum and maximum score"])
def summary_stats(df, category="Overall", year=2025, **_):
    d = _scores(df, category, year)
    if d.empty:
        return pd.DataFrame(), empty_fig("No scores are published for this category and year.")
    s = d["score"]
    t = pd.DataFrame({"statistic": ["Institutions", "Mean", "Median", "Minimum", "Maximum", "Std. deviation"],
                      "value": [len(s), s.mean(), s.median(), s.min(), s.max(), s.std()]}).round(2)
    fig = go.Figure(go.Box(x=s, name="", boxpoints="all", jitter=0.4, pointpos=0, marker=dict(color=C["brand"], size=5, opacity=0.6),
                           line=dict(color=C["ink"]), boxmean=True, text=d["institute_name"],
                           hovertemplate="%{text}<br>score %{x}<extra></extra>"))
    for lab, v, col in (("min", s.min(), C["slate"]), ("median", s.median(), C["coral"]), ("mean", s.mean(), C["amber"]), ("max", s.max(), C["slate"])):
        fig.add_vline(x=v, line_color=col, line_dash="dot", annotation_text=f"{lab} {v:.1f}", annotation_position="top")
    fig.update_xaxes(title="Overall score")
    fig.update_yaxes(visible=False)
    style(fig, f"Score summary, {category} {year}", f"n = {len(s)}; box = quartiles, dashed diamond = mean, dots = institutions.", height=380)
    return t, fig


@question("3.4", "What percentage of ranked institutions fall into each score quartile?", "distribution",
          ["category", "year"],
          ["score quartile", "quartiles", "percentage of institutions fall", "each quartile", "quartile share"])
def score_quartiles(df, category="Overall", year=2025, **_):
    d = _scores(df, category, year)
    if len(d) < 8:
        return pd.DataFrame(), empty_fig("Too few scored institutions for quartiles.")
    q = d["score"].quantile([0, .25, .5, .75, 1]).to_numpy()
    edges = np.unique(q)
    labels = [f"Q{i+1}: {edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    d = d.assign(quartile=pd.cut(d["score"], bins=edges, labels=labels, include_lowest=True))
    t = d.groupby("quartile", observed=True).size().rename("institutions").reset_index()
    t["percent"] = (100 * t["institutions"] / t["institutions"].sum()).round(1)
    fig = go.Figure(go.Bar(x=t["quartile"].astype(str), y=t["percent"], marker_color=[C["slate"], "#7E88D2", C["brand"], C["ink"]][:len(t)],
                           text=t["percent"].map("{:.1f}%".format), textposition="outside"))
    fig.update_yaxes(title="% of scored institutions", range=[0, max(35, t["percent"].max() + 8)])
    fig.update_xaxes(title="Score quartile (cut-points from the data)")
    ranked = int(scope(df, category, year)["rank_status"].isin(["exact", "band"]).sum())
    style(fig, f"Share of institutions in each score quartile, {category} {year}",
          f"Quartiles are computed on {len(d)} scored institutions (of {ranked} ranked); shares differ from 25% only through tied scores.")
    return t, fig


@question("3.5", "How does the spread of scores differ across NIRF categories?", "distribution",
          ["year"],
          ["spread of scores", "across categories", "score spread", "differ across nirf categories", "categories compare"])
def spread_by_category(df, year=2025, **_):
    d = scope(df, None, year, require_score=True)
    if d.empty:
        return pd.DataFrame(), empty_fig("No scores for this year.")
    d = d.assign(scheme=d["weight_scheme"].fillna("weighting not specified"))
    stats = d.groupby(["category", "scheme"])["score"].agg(n="count", median="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75),
                                                           minimum="min", maximum="max", std="std").reset_index()
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats = stats.sort_values("median", ascending=False)
    fig = go.Figure()
    schemes = list(dict.fromkeys(stats["scheme"]))
    for i, sch in enumerate(schemes):
        cats = stats.loc[stats["scheme"] == sch, "category"]
        sub = d[(d["scheme"] == sch)]
        fig.add_trace(go.Box(x=sub["category"] + " (n=" + sub.groupby("category")["score"].transform("count").astype(str) + ")",
                             y=sub["score"], name=sch, marker_color=SERIES[i % len(SERIES)], boxpoints="outliers"))
    fig.update_layout(boxmode="group")
    fig.update_yaxes(title="Overall score")
    style(fig, f"Score spread by category, {year}", "Colour = weighting scheme. Categories with different weights are not directly comparable.",
          legend_title="Weights (TLR/RPC/GO/OI/PR)", height=520)
    fig.update_xaxes(tickangle=-30)
    return stats.round(2), fig


# ============================================================ SECTION 4
def _scatter(df, x, y, category, year, trendline=True):
    d = scope(df, category, year)
    d = d.dropna(subset=[x, y]) if x in d and y in d else pd.DataFrame()
    if len(d) < 3:
        return pd.DataFrame(), empty_fig("Fewer than 3 institutions have both values for these filters.")
    fit = ols(d[x], d[y])
    fig = go.Figure(go.Scatter(x=d[x], y=d[y], mode="markers", text=d["institute_name"],
                               marker=dict(size=8, color=C["brand"], opacity=0.7, line=dict(color="white", width=0.6)),
                               hovertemplate="<b>%{text}</b><br>" + NAMES.get(x, x) + ": %{x:.1f}<br>" + NAMES.get(y, y) + ": %{y:.1f}<extra></extra>"))
    sub = f"n = {len(d)}"
    if fit:
        sub += f"  |  Pearson r = {fit['r']:.2f}  |  Spearman ρ = {fit['rho']:.2f}"
        if trendline:
            xs = np.array([d[x].min(), d[x].max()])
            fig.add_trace(go.Scatter(x=xs, y=fit["intercept"] + fit["slope"] * xs, mode="lines", name="Linear trend",
                                     line=dict(color=C["coral"], width=2, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(title=NAMES.get(x, x))
    fig.update_yaxes(title=NAMES.get(y, y), autorange="reversed" if y.startswith("rank") else True)
    if x.startswith("rank"):
        fig.update_xaxes(autorange="reversed")
    style(fig, f"{NAMES.get(x, x)} vs {NAMES.get(y, y)}, {category} {year}", sub)
    fig.update_layout(showlegend=False)
    out = d[["institute_name", "state", x, y]].copy()
    return out.reset_index(drop=True), fig


@question("4.1", "How strongly are Research and Professional Practice (RPC) scores associated with overall scores?", "correlation",
          ["category", "year", "trendline"],
          ["rpc and overall", "research and professional practice", "research score associated", "rpc scores associated", "rpc vs overall", "research vs overall score"])
def rpc_vs_score(df, category="Overall", year=2025, trendline=True, **_):
    return _scatter(df, "RPC", "score", category, year, trendline)


@question("4.2", "How does Perception (PR) relate to overall scores?", "correlation",
          ["category", "year", "trendline"],
          ["perception relate", "perception and overall", "pr vs overall", "perception score", "how does perception"])
def pr_vs_score(df, category="Overall", year=2025, trendline=True, **_):
    return _scatter(df, "PR", "score", category, year, trendline)


@question("4.3", "Which NIRF parameter has the strongest correlation with the overall score?", "correlation",
          ["category", "year"],
          ["strongest correlation", "which parameter correlates", "parameter has the strongest", "most correlated with overall", "correlation with the overall score"])
def strongest_parameter(df, category="Overall", year=2025, **_):
    d = scope(df, category, year).dropna(subset=["score"])
    rows = []
    for p in PARAMS:
        f = ols(d[p], d["score"]) if p in d else None
        if f:
            rows.append(dict(parameter=p, name=PARAM_NAMES[p], pearson_r=f["r"], spearman_rho=f["rho"], n=f["n"]))
    t = pd.DataFrame(rows)
    if t.empty:
        return t, empty_fig("No component scores are published for this category and year.")
    t = t.reindex(t["pearson_r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    fig = go.Figure(go.Bar(y=t["parameter"] + " · " + t["name"], x=t["pearson_r"], orientation="h",
                           marker_color=[C["brand"] if i == 0 else "#B7BDE8" for i in range(len(t))],
                           text=t["pearson_r"].map("{:.2f}".format), textposition="outside"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[min(0, t["pearson_r"].min() - 0.1), 1.05], title="Pearson correlation with overall score")
    style(fig, f"Parameters ranked by correlation with the overall score, {category} {year}",
          f"Strongest: {t.loc[0, 'parameter']} (r = {t.loc[0, 'pearson_r']:.2f}). n = {int(t['n'].max())}.", height=400)
    return t.round(3), fig


@question("4.4", "How are TLR and RPC scores related across institutions?", "correlation",
          ["category", "year", "trendline"],
          ["tlr and rpc", "teaching and research", "tlr vs rpc", "teaching related to research", "tlr rpc related"])
def tlr_vs_rpc(df, category="Overall", year=2025, trendline=True, **_):
    return _scatter(df, "TLR", "RPC", category, year, trendline)


@question("4.5", "Has the correlation between research scores and overall scores changed from 2017 to 2025?", "correlation",
          ["category", "year_from", "year_to", "parameter"],
          ["correlation changed", "correlation over time", "correlation between research", "has the correlation", "correlation trend"])
def correlation_over_time(df, category="Overall", year_from=2017, year_to=2025, parameter="RPC", **_):
    d = scope(df, category, year_from=year_from, year_to=year_to).dropna(subset=["score"])
    rows = []
    for y, g in d.groupby("year"):
        for p in PARAMS:
            f = ols(g[p], g["score"])
            if f:
                rows.append(dict(year=int(y), parameter=p, pearson_r=f["r"], n=f["n"]))
    t = pd.DataFrame(rows)
    if t.empty:
        return t, empty_fig("No component scores available for this period.")
    fig = go.Figure()
    for p in PARAMS:
        s = t[t["parameter"] == p]
        sel = p == parameter
        fig.add_trace(go.Scatter(x=s["year"], y=s["pearson_r"], mode="lines+markers", name=p, line=dict(color=PARAM_COLORS[p], width=4 if sel else 1.5),
                                 opacity=1 if sel else 0.45, marker=dict(size=9 if sel else 5), customdata=s["n"],
                                 hovertemplate=f"{p} %{{x}}: r=%{{y:.2f}} (n=%{{customdata}})<extra></extra>"))
    fig.update_xaxes(dtick=1, title="Year")
    fig.update_yaxes(title="Pearson r with overall score", range=[-0.2, 1.05])
    style(fig, f"Correlation of each parameter with the overall score, {category}",
          f"{parameter} highlighted. Correlation is computed within each year, so weighting changes do not distort it.")
    return t.round(3), fig


@question("4.6", "Explore any two numeric variables (scatter, correlation coefficient, trend line).", "correlation",
          ["category", "year", "x", "y", "trendline"],
          ["scatter", "relationship between", "plot against", "scatter plot", "versus", "correlate"])
def explore_two(df, category="Overall", year=2025, x="TLR", y="RPC", trendline=True, **_):
    return _scatter(df, x, y, category, year, trendline)


# ============================================================ SECTION 6
@question("6.1", "How does the score distribution differ between the top 10 and the remaining institutions?", "distribution",
          ["category", "year"],
          ["top 10 and the remaining", "top 10 versus the rest", "top 10 vs rest", "top ten compared", "differ between the top 10", "compare with the rest", "top 10 compare", "rest of the institutions", "versus the rest"])
def top10_vs_rest(df, category="Overall", year=2025, **_):
    d = scope(df, category, year, exact_only=True, require_score=True)
    if d.empty:
        return pd.DataFrame(), empty_fig("No exact-ranked, scored institutions here.")
    d = d.assign(group=np.where(d["rank"] <= 10, "Top 10", "Rest of ranked list"))
    t = d.groupby("group")["score"].agg(n="count", mean="mean", median="median", std="std", minimum="min", maximum="max").reset_index()
    t["iqr"] = d.groupby("group")["score"].apply(iqr).values
    fig = go.Figure()
    for g, col in (("Top 10", C["brand"]), ("Rest of ranked list", C["slate"])):
        s = d[d["group"] == g]
        fig.add_trace(go.Violin(y=s["score"], name=f"{g} (n={len(s)})", box_visible=True, meanline_visible=True, points="all", jitter=0.3,
                                pointpos=0, fillcolor=col, opacity=0.55, line_color=col, text=s["institute_name"],
                                hovertemplate="%{text}<br>score %{y}<extra></extra>"))
    fig.update_yaxes(title="Overall score")
    style(fig, f"Score distribution: top 10 vs the rest, {category} {year}", "Exact-ranked institutions with a published score.")
    fig.update_layout(showlegend=False)
    return t.round(2), fig


@question("6.2", "Are NIRF scores more dispersed in one year than another?", "distribution",
          ["category", "year_from", "year_to"],
          ["more dispersed", "dispersion", "spread in one year", "scores dispersed", "variation across years"])
def dispersion_by_year(df, category="Overall", year_from=2017, year_to=2025, **_):
    d = scope(df, category, year_from=year_from, year_to=year_to, require_score=True)
    if d.empty:
        return pd.DataFrame(), empty_fig("No scores in this period.")
    t = d.groupby("year")["score"].agg(n="count", std="std", mean="mean").reset_index()
    t["iqr"] = d.groupby("year")["score"].apply(iqr).values
    t["cv_pct"] = 100 * t["std"] / t["mean"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["year"], y=t["std"], mode="lines+markers", name="Standard deviation", line=dict(color=C["brand"], width=2.5)))
    fig.add_trace(go.Scatter(x=t["year"], y=t["iqr"], mode="lines+markers", name="Interquartile range", line=dict(color=C["amber"], width=2.5)))
    mx = t.loc[t["std"].idxmax(), "year"]
    fig.update_xaxes(dtick=1, title="Year")
    fig.update_yaxes(title="Score points")
    style(fig, f"Score dispersion by year, {category}", f"Most dispersed year (by std. deviation): {int(mx)}. Check n: the number of scored institutions changed over time.")
    return t.round(2), fig


@question("6.3", "Which NIRF parameter exhibits the greatest variability across institutions?", "distribution",
          ["category", "year"],
          ["greatest variability", "most variable parameter", "parameter variability", "which parameter varies", "highest variation"])
def parameter_variability(df, category="Overall", year=2025, **_):
    d = scope(df, category, year).dropna(subset=PARAMS)
    if d.empty:
        return pd.DataFrame(), empty_fig("No component scores for this category and year.")
    rows = [dict(parameter=p, name=PARAM_NAMES[p], std=d[p].std(), mean=d[p].mean(), cv_pct=100 * d[p].std() / d[p].mean(), iqr=iqr(d[p])) for p in PARAMS]
    t = pd.DataFrame(rows).sort_values("std", ascending=False).reset_index(drop=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Standard deviation (points)", "Coefficient of variation (%)"))
    fig.add_trace(go.Bar(x=t["parameter"], y=t["std"], marker_color=[PARAM_COLORS[p] for p in t["parameter"]], text=t["std"].round(1), textposition="outside"), 1, 1)
    c = t.sort_values("cv_pct", ascending=False)
    fig.add_trace(go.Bar(x=c["parameter"], y=c["cv_pct"], marker_color=[PARAM_COLORS[p] for p in c["parameter"]], text=c["cv_pct"].round(0), textposition="outside"), 1, 2)
    fig.update_layout(showlegend=False)
    style(fig, f"Variability of the five parameters, {category} {year}",
          f"Largest spread in points: {t.loc[0, 'parameter']}; largest relative to its mean: {c.iloc[0]['parameter']}. n = {len(d)}.")
    for a in fig.layout.annotations:
        a.font = dict(size=12, color=C["slate"])
    return t.round(2), fig


@question("6.4", "How do the median and interquartile range of scores change over time?", "distribution",
          ["category", "year_from", "year_to"],
          ["median and interquartile", "median and iqr", "iqr change over time", "interquartile range over time", "median score over time"])
def median_iqr_over_time(df, category="Overall", year_from=2017, year_to=2025, **_):
    d = scope(df, category, year_from=year_from, year_to=year_to, require_score=True)
    if d.empty:
        return pd.DataFrame(), empty_fig("No scores in this period.")
    d = d.assign(grp=d["score_comparison_group"].fillna("weighting not specified"))
    t = d.groupby(["year", "grp"])["score"].agg(n="count", median="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75)).reset_index()
    t["iqr"] = t["q3"] - t["q1"]
    fig = go.Figure()
    for i, (g, s) in enumerate(t.groupby("grp")):
        col = SERIES[i % len(SERIES)]
        fig.add_trace(go.Scatter(x=list(s["year"]) + list(s["year"])[::-1], y=list(s["q3"]) + list(s["q1"])[::-1], fill="toself",
                                 fillcolor="rgba(15,123,138,0.15)" if i == 0 else "rgba(217,154,0,0.15)", line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=s["year"], y=s["median"], mode="lines+markers", name=f"Median · {g}", line=dict(color=col, width=2.5),
                                 customdata=s[["q1", "q3", "n"]], hovertemplate="%{x}: median %{y:.1f}<br>IQR %{customdata[0]:.1f}–%{customdata[1]:.1f}, n=%{customdata[2]}<extra></extra>"))
    fig.update_xaxes(dtick=1, title="Year")
    fig.update_yaxes(title="Overall score")
    style(fig, f"Median score and interquartile range, {category}", "Shaded band = middle 50% of institutions; separate lines where the weighting scheme changed.")
    return t.round(2), fig


@question("6.5", "How does the score distribution differ across states?", "distribution",
          ["category", "year", "min_n"],
          ["across states", "score by state", "state score distribution", "scores differ by state", "state wise score"])
def distribution_by_state(df, category="Overall", year=2025, min_n=5, **_):
    d = scope(df, category, year, require_score=True)
    cnt = d.groupby("state")["score"].transform("count")
    d = d[cnt >= min_n]
    if d.empty:
        return pd.DataFrame(), empty_fig(f"No state has {min_n}+ scored institutions here. Lower the minimum.")
    t = d.groupby("state")["score"].agg(n="count", median="median", mean="mean", std="std", minimum="min", maximum="max").reset_index().sort_values("median", ascending=False)
    fig = go.Figure()
    for st in t["state"]:
        s = d[d["state"] == st]
        fig.add_trace(go.Box(y=s["score"], name=f"{st} ({len(s)})", marker_color=C["brand"], boxpoints="all", jitter=0.4, pointpos=0, showlegend=False))
    fig.update_yaxes(title="Overall score")
    fig.update_xaxes(tickangle=-35)
    style(fig, f"Score distribution by state, {category} {year}", f"States with at least {min_n} scored institutions; sorted by median; (n) after each name.", height=520)
    return t.round(2), fig
