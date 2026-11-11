"""Chart builder page: pick any measure, grouping and filters and the chart is drawn live."""
from __future__ import annotations

import json

import streamlit as st

from analysis.custom import AGGS, CHARTS, GROUPS, METRICS, NAME_PATTERNS, SPLITS, CustomChartError, build_chart, describe
from app import ui
from engine.data_loader import category_options, state_options, year_range


def page() -> None:
    df = ui.load_data()
    flt = ui.get_filters(df)
    y0, y1 = year_range(df)

    if "_pending_builder" in st.session_state:
        for pk, pv in st.session_state.pop("_pending_builder").items():
            st.session_state[pk] = pv

    ui.page_header("Build a chart", "Choose what to measure, how to group it and which institutions to include. The chart is drawn as you change the options, "
                                    "and the table underneath is the exact data behind it.")

    PRESETS = {
        "🏛️ IITs Score Trend": {"b_metric": "score", "b_agg": "mean", "b_chart": "line", "b_group": "year", "b_family": "IIT", "b_cat": "Engineering"},
        "🗺️ State-wise Research (RPC)": {"b_metric": "RPC", "b_agg": "mean", "b_chart": "bar", "b_group": "state", "b_cat": "Overall", "b_order": "Highest first"},
        "📦 Category Score Spread": {"b_metric": "score", "b_chart": "box", "b_group": "category", "b_split": None},
        "🔬 Teaching vs Research": {"b_metric": "TLR", "b_y": "RPC", "b_chart": "scatter", "b_group": None, "b_cat": "Engineering"},
        "🏙️ Cities with Most Colleges": {"b_metric": "count", "b_chart": "bar", "b_group": "city", "b_cat": "College", "b_order": "Highest first"},
    }

    st.markdown("**Quick Preset Templates:**")
    p_cols = st.columns(len(PRESETS))
    for idx, (p_label, p_params) in enumerate(PRESETS.items()):
        if p_cols[idx].button(p_label, key=f"btn_preset_{idx}", width="stretch"):
            st.session_state["_pending_builder"] = p_params
            st.rerun()
    cats = category_options(df)
    with st.container(border=True):
        a, b, c = st.columns(3)
        metric = a.selectbox("Measure", list(METRICS), format_func=METRICS.get, key="b_metric")
        agg = b.selectbox("Combine rows by", list(AGGS), format_func=AGGS.get, key="b_agg", disabled=metric == "count",
                          help="How the rows in each group are combined. A count always counts distinct institutions.")
        chart = c.selectbox("Chart type", CHARTS, format_func=lambda v: "Pick for me" if v == "auto" else v.capitalize(), key="b_chart")
        d, e, f = st.columns(3)
        group = d.selectbox("Group by", [None, *GROUPS], format_func=lambda v: "No grouping" if v is None else v.capitalize(), key="b_group")
        split = e.selectbox("Split by", [None, *SPLITS], format_func=lambda v: "No split" if v is None else v.capitalize(), key="b_split",
                            help="Adds a second dimension: one line or bar series per value, or the rows of a heatmap.")
        metric_y = f.selectbox("Second measure (scatter plot)", [None, *[m for m in METRICS if m != "count"]], key="b_y",
                               format_func=lambda v: "None" if v is None else METRICS[v])
    with st.container(border=True):
        g1, g2 = st.columns(2)
        multi_cat = group == "category" or split == "category"
        if multi_cat:
            categories = g1.multiselect("Categories", cats, default=["Engineering", "University", "Management"], key="b_cats")
            category = None
        else:
            category = g1.selectbox("Category", cats, index=cats.index(flt["category"]) if flt["category"] in cats else 0, key="b_cat")
            categories = []
        rng = g2.slider("Years", y0, y1, (flt["year_from"], flt["year_to"]), key="b_years")
        h1, h2, h3, h4 = st.columns(4)
        states = h1.multiselect("States (all if empty)", state_options(df), key="b_states")
        family = h2.selectbox("Institution family", [None, *NAME_PATTERNS], format_func=lambda v: "All institutions" if v is None else v, key="b_family")
        rank_max = h3.number_input("Only ranks up to (0 = all)", 0, 500, 0, step=10, key="b_rankmax")
        top_n = h4.slider("Show at most", 3, 50, 15, key="b_topn", help="Applies to bars and boxes grouped by state, city, category or institution.")
        i1, i2 = st.columns(2)
        exact = i1.toggle("Exact published ranks only", value=True, key="b_exact")
        order = i2.radio("Order", ["Default", "Highest first", "Lowest first"], horizontal=True, key="b_order")
        # Action buttons: generate chart / reset
        run_col, reset_col = st.columns([1, 1])
        if run_col.button("Generate chart", key="btn_generate"):
            st.session_state["_run_builder"] = True
        if reset_col.button("Reset to defaults", key="btn_reset"):
            for k in [
                "b_metric", "b_agg", "b_chart", "b_group", "b_split", "b_y",
                "b_cats", "b_cat", "b_years", "b_states", "b_family",
                "b_rankmax", "b_topn", "b_exact", "b_order",
                "_run_builder", "_pending_builder",
            ]:
                st.session_state.pop(k, None)
            st.rerun()

    spec = {"metric": metric, "agg": agg if metric != "count" else None, "group_by": group, "split_by": split, "chart": chart, "metric_y": metric_y,
            "category": category, "categories": categories or None, "states": states or None, "name_contains": family,
            "year_from": rng[0], "year_to": rng[1], "rank_max": rank_max or None, "exact_only": exact, "top_n": top_n,
            "sort": {"Highest first": "desc", "Lowest first": "asc"}.get(order)}
    try:
        data, fig, notes, clean = build_chart(df, spec)
    except CustomChartError as err:
        st.warning(str(err))
        return
    st.markdown(f"### {describe(clean)[0].upper() + describe(clean)[1:]}")
    st.plotly_chart(fig, width="stretch", key="builder_fig", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    for n in notes:
        st.caption(n)
    with st.expander(f"Data behind this chart ({len(data):,} rows)"):
        st.dataframe(data, hide_index=True, width="stretch")
        st.download_button("Download CSV", data.to_csv(index=False).encode(), file_name="nirf_custom_chart.csv", mime="text/csv", key="b_dl")
    with st.expander("Chart request (JSON)"):
        st.code(json.dumps({k: v for k, v in clean.items()}, indent=2, default=str), language="json")
