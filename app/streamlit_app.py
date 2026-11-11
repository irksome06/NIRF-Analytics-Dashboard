"""NIRF dashboard entry point.   Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
from pathlib import Path
import os
import streamlit as st

# Set API key
os.environ["GEMINI_API_KEY"] = "API_KEY"

# Page configuration must be set before any UI elements are rendered
st.set_page_config(page_title="NIRF Rankings Explorer", page_icon=":material/school:", layout="wide", initial_sidebar_state="expanded")

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import ui  # noqa: E402
from app.views import about, builder, chat, institutions, overview, sections  # noqa: E402

# Load data and sidebar filters (store in session state)
df = ui.load_data()
st.session_state["flt"] = ui.sidebar_filters(df)

# Inject CSS (light theme only)
ui.inject_css()

st.logo(str(ui.WORDMARK), size="large")
st.sidebar.caption("Build: full-names-fix 2026-11-07")

SECTION_META = {
    1: ("Trends", "show_chart", "trends"),
    2: ("Geography", "map", "geography"),
    3: ("Distributions", "bar_chart", "distributions"),
    4: ("Relationships", "scatter_plot", "relationships"),
    5: ("Parameter profiles", "radar", "profiles"),
    6: ("Statistical comparison", "stacked_bar_chart", "comparison"),
    7: ("Rank mobility", "swap_vert", "mobility"),
    8: ("Outliers and gaps", "troubleshoot", "outliers"),
}
ui.SECTION_PAGES = {s: st.Page(sections.PAGES[s], title=t, icon=f":material/{i}:", url_path=u) for s, (t, i, u) in SECTION_META.items()}
S = ui.SECTION_PAGES
PAGES = {
    "Explore": [
        st.Page(overview.page, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(institutions.page, title="Institutions", icon=":material/account_balance:", url_path="institutions"),
        *[S[k] for k in range(1, 9)],
        st.Page(builder.page, title="Build a chart", icon=":material/tune:", url_path="builder"),
    ],
    "Assistant": [st.Page(chat.page, title="Ask NIRF", icon=":material/chat:", url_path="ask")],
    "About": [st.Page(about.page, title="Data and methods", icon=":material/info:", url_path="methods")],
}

st.navigation(PAGES).run()