"""Overview page: headline numbers, top institutions, trajectories and a map."""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import ui
from engine.helpers import short_name, C, SERIES, scope, style
from engine.registry import QUESTIONS, SECTION_TITLES, by_section
from engine.query_engine import run_question


def page() -> None:
    df = ui.load_data()
    flt = ui.get_filters(df)
    cat, year = flt["category"], flt["year"]
    ui.page_header("NIRF rankings explorer",
                   "Nine editions of the National Institutional Ranking Framework (2017–2025) across 16 categories. "
                   "Choose a category and year in the sidebar, then open an analysis from the menu or ask a question in plain English.")

    QUICK_CATS = ["Overall", "Engineering", "University", "Management", "College", "Pharmacy"]
    qc_cols = st.columns(len(QUICK_CATS))
    for idx, qcat in enumerate(QUICK_CATS):
        if qc_cols[idx].button(qcat, key=f"ov_qc_{qcat}", width="stretch", type="primary" if qcat == cat else "secondary"):
            st.session_state["_pending_category"] = qcat
            st.rerun()
    d = scope(df, cat, year)
    ranked = d[d["rank_status"].isin(["exact", "band"])]
    scored = d.dropna(subset=["score"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ranked institutions", f"{len(ranked):,}", help=f"{int((d['rank_status'] == 'exact').sum())} exact ranks and {int((d['rank_status'] == 'band').sum())} in published bands.")
    c2.metric("Median score", f"{scored['score'].median():.1f}" if len(scored) else "n/a", help=f"{len(scored)} institutions have a published score.")
    c3.metric("States & UTs", f"{ranked['state'].nunique()}")
    c4.metric("Highest score", f"{scored['score'].max():.1f}" if len(scored) else "n/a", help=f"Top score in {cat}, {year}.")

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown(f"### Top 10, {cat} {year}")
        t = d[d["rank_status"].eq("exact")].sort_values("rank").head(10)
        rows = "".join(
            f"<div class='lb-row'><div class='lb-rank'>{int(r['rank'])}</div>"
            f"<div><div class='lb-name'>{html.escape(r['institute_name'])}</div>"
            f"<div class='lb-sub'>{html.escape(str(r['city']))}, {html.escape(str(r['state']))}</div>"
            + (f"<div class='lb-bar'><i style='width:{max(0, min(100, r['score'])):.0f}%'></i></div>" if pd.notna(r["score"]) else "")
            + f"</div><div class='lb-score'>{'' if pd.isna(r['score']) else format(r['score'], '.1f')}<small>score</small></div></div>"
            for _, r in t.iterrows())
        st.markdown(f"<div class='lb'>{rows}</div>", unsafe_allow_html=True)
    with right:
        lo, hi = flt["year_from"], flt["year_to"]
        top5 = d[d["rank_status"].eq("exact")].sort_values("rank").head(5)["institution_key"].tolist()
        h = scope(df, cat, year_from=lo, year_to=hi, exact_only=True)
        h = h[h["institution_key"].isin(top5)]
        fig = go.Figure()
        for i, k in enumerate(top5):
            s = h[h["institution_key"].eq(k)].sort_values("year")
            if not s.empty:
                fig.add_trace(go.Scatter(x=s["year"], y=s["rank"], mode="lines+markers", name=s["institute_name"].iloc[0],
                                         line=dict(color=SERIES[i % len(SERIES)], width=2.5), marker=dict(size=7)))
        fig.update_yaxes(autorange="reversed", title="Rank", dtick=1)
        fig.update_xaxes(dtick=1)
        style(fig, f"Rank paths of the {cat} top 5 ({year})", "Exact published ranks only.", height=380)
        fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0, yanchor="top", xanchor="left"), margin=dict(b=100))
        st.plotly_chart(fig, width="stretch", key="ov_paths", config={"displaylogo": False})

    st.markdown("### Where institutions are")
    res = run_question(df, "2.6", category=cat, year=year, map_metric="Institutions in top 100", map_style="Boundary map (offline)")
    res.figure.update_layout(height=520)
    st.plotly_chart(res.figure, width="stretch", key="ov_map", config={"displaylogo": False})

    st.markdown("### Keep exploring")
    cols = st.columns(4)
    for i, sec in enumerate(sorted(SECTION_TITLES)):
        page = ui.SECTION_PAGES.get(sec)
        if page is not None:
            cols[i % 4].page_link(page, label=f"{SECTION_TITLES[sec]} ({len(by_section(sec))})", use_container_width=True)
