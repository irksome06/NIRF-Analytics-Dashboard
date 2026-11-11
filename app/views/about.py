"""Data & methods page: coverage, definitions and caveats."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import ui
from engine.data_loader import DATA_NOTES, PARAM_NAMES, coverage_summary

DATA_DICTIONARY = [
    ("institution_key", "string", "Canonical cross-year unique institution identifier"),
    ("institute_name", "string", "Standardized institutional title"),
    ("institute_id", "string", "Official NIRF institution registration code"),
    ("institute_name_original", "string", "Verbatim raw name as published by NIRF"),
    ("year", "integer", "NIRF ranking edition year (2017–2025)"),
    ("category", "string", "Standardized ranking discipline / category"),
    ("category_original", "string", "Raw category designation from source list"),
    ("category_group", "string", "High-level classification group"),
    ("city", "string", "Standardized municipality / city"),
    ("city_original", "string", "Original raw city designation"),
    ("state", "string", "Standardized State or Union Territory"),
    ("state_code", "string", "Standard 2-letter postal abbreviation"),
    ("state_original", "string", "Original raw state text from source table"),
    ("rank", "float", "Exact published integer rank (null for banded institutions)"),
    ("rank_value", "float", "Analytical rank: exact rank or band midpoint estimate"),
    ("rank_status", "string", "Rank classification ('exact', 'band', 'not_published')"),
    ("rank_original", "string", "Published rank string (e.g. '1', '101-150')"),
    ("rank_low", "float", "Lower bound for banded ranks (e.g. 101)"),
    ("rank_high", "float", "Upper bound for banded ranks (e.g. 150)"),
    ("rank_midpoint", "float", "Arithmetic midpoint of rank band (e.g. 125.5)"),
    ("is_rank_band", "boolean", "True if institution is published within a band"),
    ("rank_band_label", "string", "Formatted band display label"),
    ("score", "float", "Composite overall NIRF score (0–100 scale)"),
    ("TLR", "float", "Teaching, Learning & Resources component score (0–100)"),
    ("RPC", "float", "Research & Professional Practice component score (0–100)"),
    ("GO", "float", "Graduation Outcomes component score (0–100)"),
    ("OI", "float", "Outreach & Inclusivity component score (0–100)"),
    ("PR", "float", "Perception component score (0–100)"),
    ("weight_scheme", "string", "Inferred parameter weighting distribution (TLR/RPC/GO/OI/PR)"),
    ("score_comparison_group", "string", "Unique key isolating rows with identical weighting methodology"),
    ("is_top10", "boolean", "Indicator flag: institution placed in top 10"),
    ("is_top50", "boolean", "Indicator flag: institution placed in top 50"),
    ("is_top100", "boolean", "Indicator flag: institution placed in top 100"),
    ("is_new_entry", "boolean", "True if institution debuts in this category"),
    ("first_year_in_category", "integer", "Earliest observed year for this institution in this category"),
    ("category_first_year", "integer", "Inaugural year when NIRF first introduced this category"),
    ("categories_in_year", "integer", "Total ranking categories active in that edition"),
    ("source_page", "string", "Official NIRF source listing URL or identifier"),
    ("has_score", "boolean", "True if an audited overall score is published"),
    ("has_components", "boolean", "True if all 5 individual parameter scores exist"),
]

CATEGORY_TIMELINE = [
    (2017, "6 Categories (Inaugural)", "Overall, Engineering, Management, Pharmacy, University, College"),
    (2018, "9 Categories (+3 added)", "Law, Medical, Architecture"),
    (2020, "10 Categories (+1 added)", "Dental"),
    (2021, "11 Categories (+1 added)", "Research Institutions"),
    (2023, "13 Categories (+2 added)", "Agriculture & Allied Sectors, Innovation"),
    (2024, "16 Categories (+3 added)", "Open University, Skill University, State Public University"),
]


def page() -> None:
    df = ui.load_data()
    ui.page_header("Data and methods", "Comprehensive documentation of the master dataset, treatment of ranks and scores, category evolution, and architectural contracts.")
    cov = coverage_summary(df)
    a, b, c, d = st.columns(4)
    a.metric("Total Records", f"{cov['rows']:,}")
    b.metric("Unique Institutions", f"{cov['institutions']:,}")
    c.metric("Years Spanned", f"{cov['years']} (2017–2025)")
    d.metric("Ranking Categories", f"{df['category'].nunique()}")

    tab_summary, tab_dict, tab_evolution = st.tabs(["Methodology & Integrity", "Data Dictionary (42 Columns)", "Category Evolution Timeline"])

    with tab_summary:
        st.markdown("### Institutions listed per category and year")
        pv = df.pivot_table(index="category", columns="year", values="institution_key", aggfunc="count", fill_value=0)
        st.dataframe(pv, width="stretch")

        st.markdown("### Weighting schemes")
        st.caption("Scores are comparable across categories only when they share a scheme (order: TLR / RPC / GO / OI / PR).")
        sch = (df.assign(weight_scheme=df["weight_scheme"].fillna("not specified")).groupby("weight_scheme")["category"]
               .agg(lambda x: ", ".join(sorted(x.unique()))).rename("categories").reset_index())
        st.dataframe(sch, hide_index=True, width="stretch")
        st.download_button("Download the master dataset (CSV)", df.to_csv(index=False).encode("utf-8"), "nirf_master_final.csv", "text/csv", key="dl_master")

        st.markdown("### How ranks and scores are treated")
        st.markdown(
            "- **Exact ranks** are published positions. **Bands** (for example 101–150) are kept as published and, where a single number is needed, "
            "shown as a midpoint *estimate* that is always labelled.\n"
            "- **Scores** are compared only within the same category and weighting scheme. Charts that span a change in weights draw separate lines.\n"
            "- **Weighting schemes** were inferred from the data for most rows; treat cross-scheme score comparisons as indicative.\n"
            "- **Coverage**: exact ranks and component scores exist only for the upper part of each list, so score-based analyses describe that group.\n"
            "- **New entries** are institutions appearing in a category for the first time after that category's first edition.\n"
            "- **Outliers** use standard scores relative to peers in the same category and year; thresholds are adjustable in each analysis."
        )
        st.markdown("### The five NIRF parameters")
        st.table({"Code": list(PARAM_NAMES), "Parameter": list(PARAM_NAMES.values())})
        st.markdown("### Architecture & Safety Guarantee")
        st.markdown("Analysis modules return `(result_table, plotly_figure)`. They are registered once and used by the pages, the query validator and the chatbot, "
                    "so a question behaves the same everywhere. The Gemini fallback only translates a question into a structured query; it never produces code that is run.")
        with st.expander("Data notes"):
            for v in DATA_NOTES.values():
                st.write("• " + v)

    with tab_dict:
        st.markdown("### Master Dataset Schema (42 Standardized Columns)")
        st.caption("All columns present in `data/nirf_master_final.csv` with their storage types, definitions, and missing value handling.")
        dict_df = pd.DataFrame(DATA_DICTIONARY, columns=["Column Name", "Data Type", "Analytical Description"])
        dict_df["Non-Null Records"] = [f"{df[col].notna().sum():,} ({df[col].notna().mean()*100:.1f}%)" if col in df.columns else "N/A" for col in dict_df["Column Name"]]
        st.dataframe(dict_df, hide_index=True, width="stretch")

    with tab_evolution:
        st.markdown("### NIRF Category Expansion (2017 to 2025)")
        st.caption("Addressing NIRF Specification Item #3: tracking the expansion of disciplines from 6 in 2017 to 16 by 2025.")
        evo_df = pd.DataFrame(CATEGORY_TIMELINE, columns=["Year Introduced", "Milestone / Scope", "Categories Active / Added"])
        st.dataframe(evo_df, hide_index=True, width="stretch")
