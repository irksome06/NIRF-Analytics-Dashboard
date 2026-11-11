"""M5 - Multivariate: Section 5 (parameter comparisons) and Section 8 (outliers & performance gaps)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from engine.data_loader import PARAM_NAMES, PARAMS
from engine.helpers import name_labels, C, PARAM_COLORS, SERIES, empty_fig, ols, robust_z, scope, style
from engine.registry import question


def _comp(df, category, year, exact_only=True):
    """Institutions with all five component scores (published for exact-ranked rows)."""
    return scope(df, category, year, exact_only=exact_only).dropna(subset=PARAMS + ["score"])


# ============================================================ SECTION 5
@question("5.1", "How do the top five institutions compare across all five NIRF parameters?", "compare",
          ["category", "year", "top_k"],
          ["top five institutions", "top 5 institutions", "compare across all five", "five parameters", "compare the top", "radar"])
def top_compare(df, category="Overall", year=2025, top_k=5, **_):
    d = _comp(df, category, year).sort_values("rank").head(top_k)
    if d.empty:
        return pd.DataFrame(), empty_fig("No component scores are published for this category and year.")
    fig = go.Figure()
    for i, (_, r) in enumerate(d.iterrows()):
        vals = [r[p] for p in PARAMS]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=[f"{p}" for p in PARAMS] + [PARAMS[0]], name=f"#{int(r['rank'])} {r['institute_name']}",
                                      line=dict(color=SERIES[i % len(SERIES)], width=2), fill="toself", opacity=0.55,
                                      hovertemplate="%{theta}: %{r:.1f}<extra>" + r["institute_name"] + "</extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor="#E3E4EC"), angularaxis=dict(gridcolor="#E3E4EC")),
                      legend=dict(orientation="v", x=1.02, y=0.5, yanchor="middle", xanchor="left"))
    style(fig, f"Top {len(d)} institutions across the five parameters, {category} {year}",
          "TLR teaching · RPC research · GO graduation outcomes · OI outreach · PR perception", height=520)
    fig.update_layout(margin=dict(r=260))
    return d[["rank", "institute_name", "score"] + PARAMS].reset_index(drop=True), fig


def _hi_lo(df, category, year, hi, lo, hi_q, lo_q, hi_name, lo_name):
    d = _comp(df, category, year)
    if len(d) < 8:
        return pd.DataFrame(), empty_fig("Too few institutions with component scores.")
    hi_cut, lo_cut = d[hi].quantile(hi_q), d[lo].quantile(lo_q)
    flag = (d[hi] >= hi_cut) & (d[lo] <= lo_cut)
    d = d.assign(gap=d[hi] - d[lo], flagged=flag)
    fig = go.Figure()
    o = d[~d["flagged"]]
    f = d[d["flagged"]]
    fig.add_trace(go.Scatter(x=o[lo], y=o[hi], mode="markers", name="Other institutions", text=o["institute_name"],
                             marker=dict(size=7, color="#B4B8CC", opacity=0.8), hovertemplate="%{text}<br>%{x:.1f}, %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=f[lo], y=f[hi], mode="markers", name=f"High {hi_name}, low {lo_name}", customdata=f["institute_name"],
                              marker=dict(size=10, color=C["coral"], line=dict(color="white", width=1)),
                             hovertemplate="%{customdata}<br>%{x:.1f}, %{y:.1f}<extra></extra>"))
    fig.add_hline(y=hi_cut, line_dash="dot", line_color=C["slate"], annotation_text=f"{hi} ≥ {hi_cut:.1f}")
    fig.add_vline(x=lo_cut, line_dash="dot", line_color=C["slate"], annotation_text=f"{lo} ≤ {lo_cut:.1f}", annotation_position="top right")
    fig.update_xaxes(title=f"{lo_name} ({lo})")
    fig.update_yaxes(title=f"{hi_name} ({hi})")
    style(fig, f"{len(f)} institution{"s" if len(f) != 1 else ""} with high {hi_name.lower()} but low {lo_name.lower()}, {category} {year}",
          f"High = top {int(round((1 - hi_q) * 100))}% on {hi}; low = bottom {int(round(lo_q * 100))}% on {lo}. Cut-offs are relative to this category and year.", height=520)
    name_labels(fig, f[lo], f[hi], f["institute_name"], d[lo], d[hi])
    out = f.sort_values("gap", ascending=False)[["rank", "institute_name", "state", hi, lo, "gap"]].reset_index(drop=True)
    return out, fig


@question("5.2", "Which institutions have high research scores but relatively low teaching scores?", "gap",
          ["category", "year", "hi_q", "lo_q"],
          ["high research scores", "high research but low teaching", "research high teaching low", "strong research weak teaching", "high research low teaching"])
def high_research_low_teaching(df, category="Overall", year=2025, hi_q=0.75, lo_q=0.5, **_):
    return _hi_lo(df, category, year, "RPC", "TLR", hi_q, lo_q, "Research", "Teaching")


@question("5.3", "Which institutions have high teaching scores but relatively low perception scores?", "gap",
          ["category", "year", "hi_q", "lo_q"],
          ["high teaching scores", "high teaching but low perception", "teaching high perception low", "strong teaching weak perception", "high teaching low perception"])
def high_teaching_low_perception(df, category="Overall", year=2025, hi_q=0.75, lo_q=0.5, **_):
    return _hi_lo(df, category, year, "TLR", "PR", hi_q, lo_q, "Teaching", "Perception")


@question("5.4", "Which institutions have improved their overall scores despite declining scores in one or more parameters?", "gap",
          ["category", "year_from", "year_to", "min_decline", "top_n"],
          ["improved their overall scores despite", "score improved despite decline", "improved despite declining", "despite declining parameter", "overall up but parameter down"])
def improved_despite_decline(df, category="Overall", year_from=2017, year_to=2025, min_decline=2.0, top_n=15, **_):
    if int(year_from) >= int(year_to):
        return pd.DataFrame(), empty_fig("Choose a later 'to' year than 'from' year.")
    a = _comp(df, category, year_from)
    b = _comp(df, category, year_to)
    keep = ["institution_key", "score", "score_comparison_group"] + PARAMS
    m = a[keep].merge(b[keep + ["institute_name"]], on="institution_key", suffixes=("_a", "_b"))
    if m.empty:
        return m, empty_fig("No institution has component scores in both years.")
    if m["score_comparison_group_a"].ne(m["score_comparison_group_b"]).all():
        return pd.DataFrame(), empty_fig("The two years use different weighting schemes, so overall scores are not comparable. Choose years within one scheme.")
    m = m[m["score_comparison_group_a"].eq(m["score_comparison_group_b"])].copy()
    m["score_change"] = m["score_b"] - m["score_a"]
    dec = {p: m[f"{p}_b"] - m[f"{p}_a"] for p in PARAMS}
    m["declined"] = [", ".join(f"{p} ({dec[p].iloc[i]:+.1f})" for p in PARAMS if dec[p].iloc[i] <= -min_decline) for i in range(len(m))]
    m["n_declines"] = [sum(dec[p].iloc[i] <= -min_decline for p in PARAMS) for i in range(len(m))]
    sel = m[(m["score_change"] > 0) & (m["n_declines"] > 0)].sort_values("score_change", ascending=False)
    if sel.empty:
        return sel, empty_fig(f"No institution improved its score while a parameter fell by {min_decline:g}+ points.")
    sh = sel.head(top_n).sort_values("score_change")
    fig = go.Figure(go.Bar(y=sh["institute_name"].map(lambda n: n), x=sh["score_change"], orientation="h", marker_color=C["brand"],
                           customdata=sh["declined"], text=sh["score_change"].map("{:+.1f}".format), textposition="outside",
                           hovertemplate="%{y}<br>score %{x:+.1f}<br>declined: %{customdata}<extra></extra>"))
    fig.update_xaxes(title=f"Overall score change, {year_from} → {year_to}")
    style(fig, f"{len(sel)} institution{"s" if len(sel) != 1 else ""} improved overall while a parameter fell {min_decline:g}+ points, {category}",
          "Showing the largest overall gains; hover for the parameters that declined. Same weighting scheme in both years.", height=max(400, 30 * len(sh) + 170))
    out = sel[["institute_name", "score_a", "score_b", "score_change", "n_declines", "declined"]].rename(columns={"score_a": f"score_{year_from}", "score_b": f"score_{year_to}"})
    return out.round(2).reset_index(drop=True), fig


@question("5.5", "Which institutions have the largest difference between their strongest and weakest parameters?", "gap",
          ["category", "year", "top_n", "standardize"],
          ["largest difference between their strongest", "strongest and weakest parameters", "strongest vs weakest", "largest gap between parameters", "biggest parameter gap"])
def parameter_gaps(df, category="Overall", year=2025, top_n=10, standardize=False, **_):
    d = _comp(df, category, year)
    if d.empty:
        return pd.DataFrame(), empty_fig("No component scores for this category and year.")
    vals = d[PARAMS]
    use = (vals - vals.mean()) / vals.std() if standardize else vals
    d = d.assign(strongest=use.idxmax(axis=1), weakest=use.idxmin(axis=1), gap=use.max(axis=1) - use.min(axis=1),
                 strongest_value=vals.max(axis=1), weakest_value=vals.min(axis=1))
    if not standardize:
        d["strongest_value"] = [r[r["strongest"]] for _, r in d.iterrows()]
        d["weakest_value"] = [r[r["weakest"]] for _, r in d.iterrows()]
    top = d.nlargest(top_n, "gap").sort_values("gap")
    fig = go.Figure(go.Bar(y=top["institute_name"].map(lambda n: n), x=top["gap"], orientation="h", marker_color=C["brand"],
                           text=top["strongest"] + " − " + top["weakest"], textposition="inside", insidetextanchor="start",
                           textfont=dict(color="white"), customdata=top[["strongest_value", "weakest_value", "rank"]],
                           hovertemplate="%{y}<br>rank %{customdata[2]:.0f}<br>gap %{x:.1f}<extra></extra>"))
    fig.update_xaxes(title="Gap in standard deviations" if standardize else "Gap in score points")
    style(fig, f"Largest strongest-to-weakest parameter gaps, {category} {year}",
          "Label shows strongest − weakest parameter." + (" Standardised within this category and year." if standardize else ""), height=max(400, 32 * len(top) + 170))
    out = top.sort_values("gap", ascending=False)[["rank", "institute_name", "strongest", "strongest_value", "weakest", "weakest_value", "gap"]]
    return out.round(2).reset_index(drop=True), fig


# ============================================================ SECTION 8
@question("8.1", "Which institutions have unusually high research scores relative to their overall scores?", "gap",
          ["category", "year", "z"],
          ["outlier", "outliers", "unusually high research", "research scores relative to overall", "research outliers", "unusually high rpc", "research unusually high"])
def research_outliers(df, category="Overall", year=2025, z=2.0, **_):
    d = _comp(df, category, year)
    fit = ols(d["score"], d["RPC"]) if len(d) >= 8 else None
    if not fit:
        return pd.DataFrame(), empty_fig("Too few institutions with component scores.")
    d = d.assign(expected=fit["intercept"] + fit["slope"] * d["score"])
    d["residual"] = d["RPC"] - d["expected"]
    d["z"] = d["residual"] / d["residual"].std()
    flag = d["z"] >= z
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.loc[~flag, "score"], y=d.loc[~flag, "RPC"], mode="markers", name="Within expected range", text=d.loc[~flag, "institute_name"],
                             marker=dict(size=7, color="#B4B8CC"), hovertemplate="%{text}<br>score %{x:.1f}, RPC %{y:.1f}<extra></extra>"))
    f = d[flag]
    fig.add_trace(go.Scatter(x=f["score"], y=f["RPC"], mode="markers", name=f"Unusually high RPC (z ≥ {z:g})", customdata=f["institute_name"],
                              marker=dict(size=10, color=C["coral"], line=dict(color="white", width=1)),
                             hovertemplate="%{customdata}<br>score %{x:.1f}, RPC %{y:.1f}<extra></extra>"))
    xs = np.array([d["score"].min(), d["score"].max()])
    fig.add_trace(go.Scatter(x=xs, y=fit["intercept"] + fit["slope"] * xs, mode="lines", name="Expected RPC for the score", line=dict(color=C["brand"], dash="dash")))
    fig.update_xaxes(title="Overall score")
    fig.update_yaxes(title="Research & Professional Practice (RPC)")
    style(fig, f"{len(f)} institution{"s" if len(f) != 1 else ""} with unusually high research relative to overall score, {category} {year}",
          "Outlier = RPC more than z standard deviations above what a linear fit on the overall score predicts.", height=520)
    name_labels(fig, f["score"], f["RPC"], f["institute_name"], d["score"], d["RPC"])
    out = f.sort_values("z", ascending=False)[["rank", "institute_name", "state", "score", "RPC", "expected", "z"]].round(2)
    return out.reset_index(drop=True), fig


@question("8.2", "Which institutions perform substantially better in research than in teaching?", "gap",
          ["category", "year", "top_n"],
          ["better in research than in teaching", "research better than teaching", "research stronger than teaching", "research minus teaching", "substantially better in research"])
def research_over_teaching(df, category="Overall", year=2025, top_n=10, **_):
    d = _comp(df, category, year)
    if d.empty:
        return pd.DataFrame(), empty_fig("No component scores for this category and year.")
    d = d.assign(gap=d["RPC"] - d["TLR"])
    top = d.nlargest(top_n, "gap").sort_values("gap")
    top = top[top["gap"] > 0]
    if top.empty:
        return top, empty_fig("No institution has research above teaching here.")
    fig = go.Figure(go.Bar(y=top["institute_name"].map(lambda n: n), x=top["gap"], orientation="h", marker_color=C["brand"],
                           text=top["gap"].map("{:+.1f}".format), textposition="outside", customdata=top[["RPC", "TLR", "rank"]],
                           hovertemplate="%{y}<br>RPC %{customdata[0]:.1f} vs TLR %{customdata[1]:.1f}<extra></extra>"))
    fig.update_xaxes(title="RPC − TLR (points)")
    style(fig, f"Largest research-over-teaching gaps, {category} {year}", f"Across {len(d)} institutions the median gap is {d['gap'].median():+.1f}.", height=max(400, 32 * len(top) + 170))
    return top.sort_values("gap", ascending=False)[["rank", "institute_name", "state", "RPC", "TLR", "gap"]].round(2).reset_index(drop=True), fig


@question("8.3", "Which institutions experienced an unusually large year-to-year score change?", "gap",
          ["category", "year_from", "year_to", "z"],
          ["unusually large year-to-year score change", "large score change", "score jump", "score swings", "year to year score", "sudden score change"])
def score_jumps(df, category="Overall", year_from=2017, year_to=2025, z=2.5, **_):
    d = scope(df, category, year_from=year_from, year_to=year_to, require_score=True)
    rows = []
    for y in sorted(d["year"].unique())[1:]:
        a, b = d[d["year"] == y - 1], d[d["year"] == y]
        m = a[["institution_key", "score", "score_comparison_group"]].merge(b[["institution_key", "institute_name", "score", "score_comparison_group"]], on="institution_key", suffixes=("_a", "_b"))
        m = m[m["score_comparison_group_a"].eq(m["score_comparison_group_b"])]
        if len(m) < 8:
            continue
        m["change"] = m["score_b"] - m["score_a"]
        m["adjusted_change"] = m["change"] - m["change"].median()
        m["z"] = robust_z(m["change"])
        m["year"] = y
        rows.append(m)
    if not rows:
        return pd.DataFrame(), empty_fig("Not enough consecutive years with comparable scores.")
    t = pd.concat(rows)
    flag = t["z"].abs() >= z
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t.loc[~flag, "year"], y=t.loc[~flag, "adjusted_change"], mode="markers", name="Typical change", text=t.loc[~flag, "institute_name"],
                             marker=dict(size=6, color="#B4B8CC", opacity=0.6), hovertemplate="%{text}<br>%{x}: %{y:+.1f}<extra></extra>"))
    f = t[flag]
    fig.add_trace(go.Scatter(x=f["year"], y=f["adjusted_change"], mode="markers", name=f"Unusual (|z| ≥ {z:g})", text=f["institute_name"],
                             marker=dict(size=10, color=np.where(f["adjusted_change"] > 0, C["brand"], C["coral"]), line=dict(color="white", width=1)),
                             hovertemplate="%{text}<br>%{x}: %{y:+.1f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#D9DBE6")
    fig.update_xaxes(dtick=1, title="Year of change (vs previous year)")
    fig.update_yaxes(title="Score change minus that year's median change")
    style(fig, f"{len(f)} unusually large year-to-year score changes, {category}",
          "Changes are adjusted for the cohort median that year and compared only within one weighting scheme. Teal = rise, coral = fall.", height=520)
    out = f.assign(abs_z=f["z"].abs()).sort_values("abs_z", ascending=False)[["year", "institute_name", "score_a", "score_b", "change", "adjusted_change", "z"]]
    return out.rename(columns={"score_a": "previous_score", "score_b": "score"}).round(2).reset_index(drop=True), fig


@question("8.4", "Which institutions have a strong parameter score but a relatively poor overall rank?", "gap",
          ["category", "year", "parameter", "hi_q", "poor_q"],
          ["strong parameter score", "strong parameter but poor rank", "high parameter low rank", "good score poor rank", "strong in one parameter"])
def strong_param_poor_rank(df, category="Overall", year=2025, parameter="RPC", hi_q=0.75, poor_q=0.5, **_):
    d = _comp(df, category, year)
    if len(d) < 8:
        return pd.DataFrame(), empty_fig("Too few institutions with component scores.")
    cut = d[parameter].quantile(hi_q)
    rank_cut = d["rank"].quantile(poor_q)
    flag = (d[parameter] >= cut) & (d["rank"] > rank_cut)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.loc[~flag, parameter], y=d.loc[~flag, "rank"], mode="markers", name="Other institutions", text=d.loc[~flag, "institute_name"],
                             marker=dict(size=7, color="#B4B8CC"), hovertemplate="%{text}<br>" + parameter + " %{x:.1f}, rank %{y:.0f}<extra></extra>"))
    f = d[flag]
    fig.add_trace(go.Scatter(x=f[parameter], y=f["rank"], mode="markers", name="Strong parameter, weaker rank", customdata=f["institute_name"],
                              marker=dict(size=10, color=C["coral"], line=dict(color="white", width=1)),
                             hovertemplate="%{customdata}<br>" + parameter + " %{x:.1f}, rank %{y:.0f}<extra></extra>"))
    fig.add_vline(x=cut, line_dash="dot", line_color=C["slate"], annotation_text=f"{parameter} ≥ {cut:.1f}")
    fig.add_hline(y=rank_cut, line_dash="dot", line_color=C["slate"], annotation_text=f"rank > {rank_cut:.0f}", annotation_position="bottom right")
    fig.update_yaxes(autorange="reversed", title="Rank (1 = best)")
    fig.update_xaxes(title=f"{PARAM_NAMES[parameter]} ({parameter})")
    style(fig, f"{len(f)} institution{"s" if len(f) != 1 else ""} strong in {parameter} but ranked in the lower part of the list, {category} {year}",
          f"Strong = top {int(round((1 - hi_q) * 100))}% on {parameter}; poor = rank worse than the {int(round(poor_q * 100))}th percentile of ranked institutions.", height=520)
    name_labels(fig, f[parameter], f["rank"], f["institute_name"], d[parameter], d["rank"], reverse_y=True)
    return f.sort_values(parameter, ascending=False)[["rank", "institute_name", "state", parameter, "score"]].reset_index(drop=True), fig


@question("8.5", "Which institutions have unusually high ranking volatility compared with similar institutions?", "gap",
          ["category", "year_from", "year_to", "min_obs", "z", "exact_only"],
          ["ranking volatility", "volatile", "unstable rank", "rank volatility", "fluctuat", "compared with similar institutions"])
def rank_volatility(df, category="Overall", year_from=2017, year_to=2025, min_obs=4, z=2.0, exact_only=True, **_):
    col = "rank" if exact_only else "rank_value"
    d = scope(df, category, year_from=year_from, year_to=year_to, exact_only=exact_only).dropna(subset=[col])
    t = d.groupby(["institution_key", "institute_name"])[col].agg(years="count", mean_rank="mean", rank_std="std", best="min", worst="max").reset_index()
    t = t[t["years"] >= min_obs]
    if len(t) < 8:
        return t, empty_fig(f"Too few institutions are ranked in {min_obs}+ years. Lower the minimum.")
    t["peer_group"] = pd.qcut(t["mean_rank"], q=4, duplicates="drop").astype(str)
    g = t.groupby("peer_group")["rank_std"]
    t["z"] = (t["rank_std"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    flag = t["z"] >= z
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t.loc[~flag, "mean_rank"], y=t.loc[~flag, "rank_std"], mode="markers", name="Typical for peers", text=t.loc[~flag, "institute_name"],
                             marker=dict(size=7, color="#B4B8CC"), hovertemplate="%{text}<br>mean rank %{x:.0f}, std %{y:.1f}<extra></extra>"))
    f = t[flag]
    fig.add_trace(go.Scatter(x=f["mean_rank"], y=f["rank_std"], mode="markers", name=f"Unusually volatile (z ≥ {z:g})", customdata=f["institute_name"],
                              marker=dict(size=10, color=C["coral"], line=dict(color="white", width=1)),
                             hovertemplate="%{customdata}<br>mean rank %{x:.0f}, std %{y:.1f}<extra></extra>"))
    fig.update_xaxes(title="Average rank over the period (lower = better)")
    fig.update_yaxes(title="Volatility: std. deviation of rank")
    style(fig, f"{len(f)} institution{"s" if len(f) != 1 else ""} with unusually high rank volatility, {category} {year_from}–{year_to}",
          f"Compared with peers of similar average rank (quartile groups). Institutions ranked in {min_obs}+ years" + ("; exact ranks only." if exact_only else "; includes band-midpoint estimates."), height=520)
    name_labels(fig, f["mean_rank"], f["rank_std"], f["institute_name"], t["mean_rank"], t["rank_std"])
    out = f.sort_values("z", ascending=False)[["institute_name", "years", "mean_rank", "rank_std", "best", "worst", "peer_group", "z"]].round(2)
    return out.reset_index(drop=True), fig
