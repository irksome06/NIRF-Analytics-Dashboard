"""M3 - Time series & transitions: Section 1 (trends) and Section 7 (ranking mobility)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from engine.helpers import (short_name, C, ESTIMATE_NOTE, GROUP_ORDER, SERIES, empty_fig, pair_ranks, rank_col,
                            rank_group, rank_label, scope, style)
from engine.registry import question


# ============================================================ SECTION 1
@question("1.1", "How has the ranking of a selected institution changed from 2017 to 2025?", "trend",
          ["category", "institution", "year_from", "year_to"],
          ["rank trend", "ranking changed", "rank over time", "how has rank", "ranking of", "trend of", "history of"])
def rank_trend(df, category="Overall", institution=None, year_from=2017, year_to=2025, **_):
    d = scope(df, category, year_from=year_from, year_to=year_to)
    d = d[d["institution_key"].eq(institution) | d["institute_name"].eq(institution)].sort_values("year")
    if d.empty:
        return pd.DataFrame(), empty_fig("Pick an institution that is ranked in this category.")
    name = d["institute_name"].iloc[0]
    d = d.assign(rank_label=rank_label(d))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["year"], y=d["rank_value"], mode="lines", line=dict(color=C["brand"], width=2),
                             hoverinfo="skip", showlegend=False))
    ex, bd = d[d["rank_status"].eq("exact")], d[d["rank_status"].eq("band")]
    fig.add_trace(go.Scatter(x=ex["year"], y=ex["rank_value"], mode="markers+text", name="Exact rank",
                             text=ex["rank_label"], textposition="top center",
                             marker=dict(size=10, color=C["brand"]),
                             hovertemplate="%{x}: rank %{y:.0f}<extra></extra>"))
    if not bd.empty:
        fig.add_trace(go.Scatter(
            x=bd["year"], y=bd["rank_value"], mode="markers", name="Rank band (midpoint estimate)",
            marker=dict(size=10, color="white", line=dict(color=C["amber"], width=2)),
            error_y=dict(type="data", symmetric=False, array=(bd["rank_high"] - bd["rank_value"]).tolist(),
                         arrayminus=(bd["rank_value"] - bd["rank_low"]).tolist(), color=C["amber"]),
            customdata=bd["rank_label"], hovertemplate="%{x}: %{customdata}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Rank (1 = best)")
    fig.update_xaxes(dtick=1, title="Year")
    style(fig, f"{name}: rank trend, {category}", ESTIMATE_NOTE if not bd.empty else "All ranks are exact published ranks.")
    out = d[["year", "rank_label", "rank_status", "score", "state"]].reset_index(drop=True)
    return out, fig


def _movers(df, category, year_from, year_to, exact_only, top_n, improving: bool):
    m = pair_ranks(df, category, year_from, year_to, exact_only)
    if m.empty:
        return m, empty_fig("No institution is ranked in both selected years.")
    m = m[m["rank_change"] > 0] if improving else m[m["rank_change"] < 0]
    m = m.sort_values("rank_change", ascending=not improving).head(top_n)
    if m.empty:
        return m, empty_fig("No institutions " + ("improved" if improving else "declined") + " over this period.")
    m = m.assign(label=m["institute_name"].map(lambda n: n),
                 move=m["rank_change"].abs().astype(int),
                 text=lambda t: "#" + t["rank_a"].astype(int).astype(str) + " → #" + t["rank_b"].astype(int).astype(str))
    fig = go.Figure()
    for basis, color in (("Exact ranks", C["brand"] if improving else C["coral"]), ("Includes band estimate", C["brand_light"] if improving else "#EBB1A9")):
        s = m[m["basis"].eq(basis)].sort_values("move")
        if not s.empty:
            fig.add_trace(go.Bar(y=s["label"], x=s["move"], orientation="h", name=basis, marker_color=color,
                                 text=s["text"], textposition="auto", customdata=s["state"],
                                 hovertemplate="%{y}<br>%{text} (%{customdata})<extra></extra>"))
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_xaxes(title="Places gained" if improving else "Places lost")
    word = "improvement" if improving else "decline"
    style(fig, f"Largest rank {word}, {category} {year_from} → {year_to}",
          "Bars show places moved; labels show start → end rank. " + ("Light bars include band estimates." if (m['basis'] != 'Exact ranks').any() else "Exact published ranks only."),
          height=max(380, 34 * len(m) + 160))
    return m.drop(columns=["label", "text", "move"]).reset_index(drop=True), fig


@question("1.2", "Which institutions have shown the greatest improvement in rank over the nine years?", "trend",
          ["category", "year_from", "year_to", "top_n", "exact_only"],
          ["greatest improvement", "most improved", "biggest improvement", "improved the most", "climbed", "risen"])
def top_improvers(df, category="Overall", year_from=2017, year_to=2025, top_n=10, exact_only=True, **_):
    return _movers(df, category, year_from, year_to, exact_only, top_n, True)


@question("1.3", "Which institutions have experienced the largest decline in ranking?", "trend",
          ["category", "year_from", "year_to", "top_n", "exact_only"],
          ["largest decline", "biggest decline", "declined the most", "dropped the most", "fell the most", "worst decline"])
def top_decliners(df, category="Overall", year_from=2017, year_to=2025, top_n=10, exact_only=True, **_):
    return _movers(df, category, year_from, year_to, exact_only, top_n, False)


@question("1.4", "How has the average score of institutions changed over time?", "trend",
          ["category", "year_from", "year_to"],
          ["average score", "mean score", "score over time", "score trend", "average nirf score"])
def score_trend(df, category="Overall", year_from=2017, year_to=2025, **_):
    d = scope(df, category, year_from=year_from, year_to=year_to, require_score=True)
    if d.empty:
        return pd.DataFrame(), empty_fig("No scores are published for this category and period.")
    g = (d.groupby(["year", "score_comparison_group"], dropna=False)["score"]
         .agg(n="count", mean="mean", median="median").reset_index())
    g["score_comparison_group"] = g["score_comparison_group"].fillna("weighting not specified")
    fig = go.Figure()
    for i, (grp, s) in enumerate(g.groupby("score_comparison_group")):
        col = SERIES[i % len(SERIES)]
        fig.add_trace(go.Scatter(x=s["year"], y=s["mean"], mode="lines+markers", name=f"Mean · {grp}",
                                 line=dict(color=col, width=2.5), customdata=s[["n", "median"]],
                                 hovertemplate="%{x}: mean %{y:.1f}<br>median %{customdata[1]:.1f}, n=%{customdata[0]}<extra></extra>"))
        fig.add_trace(go.Scatter(x=s["year"], y=s["median"], mode="lines", name=f"Median · {grp}",
                                 line=dict(color=col, width=1.5, dash="dot"), hoverinfo="skip"))
    fig.update_xaxes(dtick=1, title="Year")
    fig.update_yaxes(title="Overall score")
    style(fig, f"Average overall score, {category}", "Scores are only comparable within a weighting scheme, so each scheme is its own line.")
    return g.round(2), fig


@question("1.5", "Which institutions have remained consistently in the top 10 across multiple years?", "trend",
          ["category", "year_from", "year_to", "min_years"],
          ["consistently in the top 10", "consistent top 10", "remained in the top 10", "always in top 10", "stayed in top 10", "top 10 across"])
def consistent_top10(df, category="Overall", year_from=2017, year_to=2025, min_years=3, **_):
    d = scope(df, category, year_from=year_from, year_to=year_to, exact_only=True)
    top = d[d["rank"] <= 10]
    cnt = top.groupby(["institution_key", "institute_name"]).agg(
        years_in_top10=("year", "nunique"), best_rank=("rank", "min"),
        years=("year", lambda s: ", ".join(map(str, sorted(s))))).reset_index()
    cnt = cnt[cnt["years_in_top10"] >= min_years].sort_values(["years_in_top10", "best_rank"], ascending=[False, True])
    if cnt.empty:
        return cnt, empty_fig(f"No institution is in the top 10 in {min_years}+ years of this period.")
    sel = d[d["institution_key"].isin(cnt["institution_key"])]
    pv = sel.pivot_table(index="institute_name", columns="year", values="rank", aggfunc="min")
    order = cnt["institute_name"].tolist()
    pv = pv.reindex(order)
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=[str(c) for c in pv.columns], y=list(pv.index),
        text=pv.map(lambda v: "" if pd.isna(v) else str(int(v))).values, texttemplate="%{text}",
        colorscale=[[0, C["brand"]], [0.12, "#6B76C9"], [0.4, "#C9CEF0"], [1, "#EFF0F7"]], zmin=1, zmax=40,
        xgap=2, ygap=2, showscale=False, hovertemplate="%{y}<br>%{x}: rank %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    style(fig, f"Institutions in the top 10 for {min_years}+ years, {category}",
          "Cell = exact rank that year; darker = better; blank = no exact rank.", height=max(360, 32 * len(pv) + 150))
    return cnt.drop(columns=["institution_key"]).reset_index(drop=True), fig


# ============================================================ SECTION 7
def _group_frame(df, category, year):
    d = scope(df, category, year)
    return pd.DataFrame({"institution_key": d["institution_key"], "institute_name": d["institute_name"],
                         "group": rank_group(d), "rank_status": d["rank_status"]})


def _transition(df, category, year_a, year_b):
    a, b = _group_frame(df, category, year_a), _group_frame(df, category, year_b)
    m = a.merge(b, on="institution_key", how="outer", suffixes=("_a", "_b"))
    m["institute_name"] = m["institute_name_a"].fillna(m["institute_name_b"])
    m["group_a"] = m["group_a"].fillna("Not in list")
    m["group_b"] = m["group_b"].fillna("Not in list")
    return m[["institution_key", "institute_name", "group_a", "group_b"]]


@question("7.1", "How many institutions moved from the 51–100 band into the top 50 between two selected years?", "transition",
          ["category", "year_a", "year_b"],
          ["51-100", "51–100", "into the top 50", "moved into top 50", "band into the top 50", "51 to 100"])
def band_to_top50(df, category="Overall", year_a=2024, year_b=2025, **_):
    m = _transition(df, category, year_a, year_b)
    movers = m[(m["group_a"] == "51–100") & m["group_b"].isin(["Top 10", "11–50"])]
    order = GROUP_ORDER + ["Not in list"]
    mat = pd.crosstab(m["group_a"], m["group_b"]).reindex(index=order, columns=order, fill_value=0)
    mat = mat.loc[(mat.sum(axis=1) > 0), (mat.sum(axis=0) > 0)]
    fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, text=mat.values, texttemplate="%{text}",
                               colorscale=[[0, "#F4F5FB"], [1, C["brand"]]], showscale=False, xgap=2, ygap=2,
                               hovertemplate=f"{year_a}: %{{y}}<br>{year_b}: %{{x}}<br>%{{z}} institutions<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=f"Rank group in {year_a}")
    fig.update_xaxes(title=f"Rank group in {year_b}", side="top")
    style(fig, f"{len(movers)} institution(s) moved from 51–100 into the top 50, {category} {year_a} → {year_b}",
          "Full transition matrix; rows = earlier year, columns = later year. Bands are used as published.", height=470)
    fig.update_layout(margin=dict(t=120))
    return movers[["institute_name", "group_a", "group_b"]].rename(
        columns={"group_a": f"group_{year_a}", "group_b": f"group_{year_b}"}).reset_index(drop=True), fig


@question("7.2", "Which institutions entered or exited the top 10 between consecutive years?", "transition",
          ["category", "year_from", "year_to"],
          ["entered the top 10", "exited the top 10", "entered or exited", "top 10 entries", "left the top 10", "joined the top 10"])
def top10_entries_exits(df, category="Overall", year_from=2017, year_to=2025, **_):
    d = scope(df, category, year_from=year_from, year_to=year_to, exact_only=True)
    sets = {y: set(g.loc[g["rank"] <= 10, "institute_name"]) for y, g in d.groupby("year")}
    years = sorted(sets)
    rows = []
    for a, b in zip(years[:-1], years[1:]):
        rows.append(dict(transition=f"{a}→{b}", entered=len(sets[b] - sets[a]), exited=len(sets[a] - sets[b]),
                         entered_names="; ".join(sorted(sets[b] - sets[a])), exited_names="; ".join(sorted(sets[a] - sets[b]))))
    out = pd.DataFrame(rows)
    if out.empty:
        return out, empty_fig("Need at least two years of exact ranks.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=out["transition"], y=out["entered"], name="Entered top 10", marker_color=C["brand"],
                         customdata=out["entered_names"], hovertemplate="%{x}<br>%{customdata}<extra></extra>"))
    fig.add_trace(go.Bar(x=out["transition"], y=-out["exited"], name="Exited top 10", marker_color=C["coral"],
                         customdata=out["exited_names"], hovertemplate="%{x}<br>%{customdata}<extra></extra>"))
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title="Institutions (exits shown below zero)")
    style(fig, f"Top-10 entries and exits, {category}", "Hover a bar for the institution names.")
    return out, fig


@question("7.3", "What percentage of institutions remained in the same rank band from one year to the next?", "transition",
          ["category", "year_a", "year_b"],
          ["same rank band", "remained in the same", "percentage of institutions remained", "stayed in the same band", "rank band stability"])
def band_stability(df, category="Overall", year_a=2024, year_b=2025, **_):
    m = _transition(df, category, year_a, year_b)
    m = m[m["group_a"] != "Not in list"]
    if m.empty:
        return m, empty_fig("No ranked institutions in the earlier year.")
    m = m.assign(same=m["group_a"].eq(m["group_b"]))
    t = m.groupby("group_a").agg(institutions=("same", "size"), retained=("same", "sum")).reset_index()
    t["pct_retained"] = (100 * t["retained"] / t["institutions"]).round(1)
    t["group_a"] = pd.Categorical(t["group_a"], GROUP_ORDER, ordered=True)
    t = t.sort_values("group_a")
    overall = 100 * m["same"].mean()
    fig = go.Figure(go.Bar(x=t["group_a"].astype(str), y=t["pct_retained"], marker_color=C["brand"],
                           text=t["pct_retained"].map("{:.0f}%".format), textposition="outside",
                           customdata=t[["retained", "institutions"]],
                           hovertemplate="%{x}: %{customdata[0]} of %{customdata[1]} stayed<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dot", line_color=C["amber"], annotation_text=f"overall {overall:.0f}%")
    fig.update_yaxes(range=[0, 112], title="% staying in the same rank group")
    fig.update_xaxes(title=f"Rank group in {year_a}")
    style(fig, f"{overall:.0f}% of institutions stayed in the same rank group, {category} {year_a} → {year_b}",
          "Institutions that left the published list count as not retained.")
    return t.rename(columns={"group_a": "rank_group"}), fig


@question("7.4", "Which institutions experienced the largest upward and downward rank movements?", "transition",
          ["category", "year_a", "year_b", "top_n", "exact_only"],
          ["largest upward", "upward and downward", "biggest movers", "rank movements", "largest movement", "biggest rank change"])
def largest_movers(df, category="Overall", year_a=2024, year_b=2025, top_n=8, exact_only=True, **_):
    m = pair_ranks(df, category, year_a, year_b, exact_only)
    if m.empty:
        return m, empty_fig("No institution is ranked in both years.")
    up = m.nlargest(top_n, "rank_change")
    down = m.nsmallest(top_n, "rank_change")
    sel = pd.concat([up[up["rank_change"] > 0], down[down["rank_change"] < 0]]).drop_duplicates("institution_key")
    sel = sel.sort_values("rank_change")
    fig = go.Figure(go.Bar(
        y=sel["institute_name"].map(lambda n: n), x=sel["rank_change"], orientation="h",
        marker_color=np.where(sel["rank_change"] > 0, C["brand"], C["coral"]),
        text=("#" + sel["rank_a"].astype(int).astype(str) + " → #" + sel["rank_b"].astype(int).astype(str)),
        textposition="auto", hovertemplate="%{y}<br>%{text}<extra></extra>"))
    fig.update_xaxes(title="Places moved (positive = improved)")
    style(fig, f"Largest rank movements, {category} {year_a} → {year_b}", "Teal = moved up, coral = moved down.",
          height=max(380, 30 * len(sel) + 160))
    return sel.reset_index(drop=True), fig


@question("7.5", "How many institutions moved up, moved down, or retained their positions each year?", "transition",
          ["category", "year_from", "year_to", "exact_only"],
          ["moved up, moved down", "up, down or retained", "retained their positions", "moved up or down each year", "how many moved up", "moved up", "moved down", "stayed each year", "up down or", "up or down"])
def movement_counts(df, category="Overall", year_from=2017, year_to=2025, exact_only=True, **_):
    rows = []
    for a in range(int(year_from), int(year_to)):
        m = pair_ranks(df, category, a, a + 1, exact_only)
        if m.empty:
            continue
        rows.append(dict(transition=f"{a}→{a+1}", moved_up=int((m.rank_change > 0).sum()),
                         moved_down=int((m.rank_change < 0).sum()), retained=int((m.rank_change == 0).sum()), compared=len(m)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out, empty_fig("No consecutive years with comparable ranks.")
    fig = go.Figure()
    for col, color, nm in (("moved_up", C["brand"], "Moved up"), ("retained", "#B4B8CC", "Retained"), ("moved_down", C["coral"], "Moved down")):
        fig.add_trace(go.Bar(x=out["transition"], y=out[col], name=nm, marker_color=color))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Institutions")
    style(fig, f"Rank movement per year, {category}",
          "Institutions ranked in both years." + ("" if exact_only else " Band midpoints are estimates."))
    return out, fig


@question("1.6", "Which institutions are ranked highest in a category and year?", "trend",
          ["category", "year", "top_n", "exact_only", "state_filter", "bottom"],
          ["top institutions", "best institutions", "highest ranked", "top ranked", "leaderboard", "ranking list", "who is ranked first", "list of the top"],
          "Leaderboard for one category and year (exact ranks by default); optionally one state, or the lowest-ranked end of the list.")
def top_institutions(df, category="Overall", year=2025, top_n=10, exact_only=True, state_filter=None, bottom=False, **_):
    d = scope(df, category, year, exact_only=exact_only)
    col = rank_col(exact_only)
    d = d[d[col].notna()]
    if state_filter:
        d = d[d["state"].eq(state_filter)]
    d = d.sort_values([col, "institute_name"], ascending=[not bottom, True]).head(int(top_n))
    if d.empty:
        where = f" in {state_filter}" if state_filter else ""
        return pd.DataFrame(), empty_fig(f"No ranked institutions{where} for this category and year.")
    d = d.assign(position=rank_label(d))
    table = d[["position", "institute_name", "state", "city", "score", "TLR", "RPC", "GO", "OI", "PR"]].reset_index(drop=True)
    sc = d.dropna(subset=["score"])
    if sc.empty:
        return table, empty_fig("This category does not publish scores. The ranking list is in the table below.")
    fig = go.Figure(go.Bar(y=sc["institute_name"], x=sc["score"], orientation="h", marker_color=C["coral"] if bottom else C["brand"],
                           text=[f"#{p}" for p in sc["position"]], textposition="inside", insidetextanchor="start", textfont=dict(color="white"),
                           hovertemplate="%{y}<br>Score %{x:.1f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="NIRF score", range=[max(0, sc["score"].min() - 8), min(100, sc["score"].max() + 3)])
    kind = "Lowest-ranked" if bottom else "Top"
    place = f", {state_filter}" if state_filter else ""
    style(fig, f"{kind} {len(table)} institutions, {category} {year}{place}", "Bars show the published score; labels show the rank.", height=max(380, 36 * len(sc) + 150))
    return table, fig


@question("1.7", "How do two institutions compare in rank over time?", "compare",
          ["category", "institution", "institution_b", "year_from", "year_to", "exact_only"],
          ["compare institutions", "compare two institutions", "institution versus institution", "head to head", "side by side"],
          "Rank paths of two institutions on one chart. If no second institution is given, the category's top-ranked institution is used.")
def compare_two(df, category="Overall", institution=None, institution_b=None, year_from=2017, year_to=2025, exact_only=True, **_):
    from analysis.institutions import compare_institutions
    if not institution_b or institution_b == institution:
        top = scope(df, category, year_to, exact_only=True).sort_values("rank")
        top = top[~top["institution_key"].eq(institution)]
        institution_b = top["institution_key"].iloc[0] if len(top) else None
    keys = [k for k in (institution, institution_b) if k]
    return compare_institutions(df, category, keys, year_from, year_to, exact_only)


@question("1.8", "How did an institution perform in a given year?", "trend",
          ["category", "institution", "year"],
          ["profile", "snapshot", "how did", "performance in", "scorecard", "overview of", "rank and score"],
          "One institution in one year: rank, score and parameter profile against the cohort.")
def institution_snapshot(df, category="Overall", institution=None, year=2025, **_):
    from analysis.institutions import institution_radar
    d = scope(df, category, year)
    row = d[d["institution_key"].eq(institution) | d["institute_name"].eq(institution)]
    if row.empty:
        name = scope(df, category)[lambda x: x["institution_key"].eq(institution) | x["institute_name"].eq(institution)]["institute_name"]
        who = name.iloc[0] if len(name) else "This institution"
        return pd.DataFrame(), empty_fig(f"{who} is not listed in {category} for {year}.")
    r = row.iloc[0]
    summary = pd.DataFrame([{"institution": r["institute_name"], "year": year, "position": rank_label(row).tolist()[0], "state": r["state"],
                             "score": r["score"], "TLR": r["TLR"], "RPC": r["RPC"], "GO": r["GO"], "OI": r["OI"], "PR": r["PR"]}])
    if r["rank_status"] == "exact" and pd.notna(r["score"]):
        _, fig = institution_radar(df, category, r["institution_key"], year)
        return summary, fig
    return summary, empty_fig(f"{r['institute_name']} appears in the {r['rank_band_label']} band in {year}; NIRF publishes no parameter scores for banded institutions.")
