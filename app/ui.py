"""Shared Streamlit UI pieces: theme, sidebar filters, generic question renderer."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from engine.data_loader import (PARAM_NAMES, category_options, coverage_summary, filter_data, institution_options, load_master,
                                state_options, year_range)
from engine.query_engine import QueryValidationError, default_params, run_question
from engine.registry import GLOBAL_PARAMS, PARAM_SPECS, QUESTIONS, SECTION_TITLES, by_section
from nlp.insights import generate_insights

SHORT = {
    "1.1": "Institution rank trend", "1.2": "Biggest improvers", "1.3": "Biggest decliners", "1.4": "Average score over time", "1.5": "Consistent top 10", "1.6": "Top institutions", "1.7": "Compare two institutions", "1.8": "Institution snapshot",
    "2.1": "Top states", "2.2": "Distribution over time", "2.3": "States gaining", "2.4": "State vs state", "2.5": "New entrants", "2.6": "India map",
    "3.1": "Score distribution", "3.2": "Rank distribution", "3.3": "Mean, median, min, max", "3.4": "Score quartiles", "3.5": "Spread by category",
    "4.1": "Research vs overall", "4.2": "Perception vs overall", "4.3": "Strongest parameter", "4.4": "Teaching vs research", "4.5": "Correlation over time", "4.6": "Explore any two variables",
    "5.1": "Top five compared", "5.2": "High research, low teaching", "5.3": "High teaching, low perception", "5.4": "Up despite parameter declines", "5.5": "Strongest vs weakest gap",
    "6.1": "Top 10 vs the rest", "6.2": "Dispersion by year", "6.3": "Parameter variability", "6.4": "Median & IQR over time", "6.5": "Distribution by state",
    "7.1": "51–100 into top 50", "7.2": "Top-10 entries & exits", "7.3": "Same rank band", "7.4": "Largest movers", "7.5": "Up / down / same",
    "8.1": "High research for score", "8.2": "Research over teaching", "8.3": "Large score changes", "8.4": "Strong parameter, poor rank", "8.5": "Rank volatility",
}
SECTION_NOTES = {
    1: "Ranks come from NIRF lists; beyond the published exact ranks NIRF reports bands (for example 101–150), shown here as estimates.",
    2: "State counts use one category at a time. An institution can appear in several categories, so counts are never summed across them.",
    3: "Scores exist only for institutions with a published exact rank. Weighting schemes differ between categories and years.",
    4: "Correlation is not causation. Parameters feed the overall score, so strong correlations are partly by construction.",
    5: "Component scores (TLR, RPC, GO, OI, PR) are published for exact-ranked institutions only.",
    6: "Statistics describe institutions with a published score; the number of scored institutions (n) is shown wherever it matters.",
    7: "Band ranks are compared as bands, not as exact positions. Institutions absent from a year's list count as 'Not in list'.",
    8: "Outliers are statistical flags relative to peers in the same category and year, not judgements about quality.",
}

CSS_PATH = Path(__file__).with_name("style.css")
WORDMARK = Path(__file__).with_name("assets") / "wordmark.svg"


SECTION_PAGES: dict = {}  # filled by streamlit_app so views can link to pages


def inject_css() -> None:
    """Inject the base CSS stylesheet.
    The dark‑mode toggle has been removed, so we only load the standard style.
    """
    try:
        base_css = CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{base_css}</style>", unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Failed to load stylesheet: {exc}")

    """Inject CSS styles with optional dark‑mode overrides.

    The light theme is defined in ``style.css``. When the user enables
    dark mode via the sidebar toggle (stored in ``st.session_state['dark_mode']``),
    ``dark_style.css`` – which only redefines the CSS colour variables –
    is injected **after** the base stylesheet so that the variable values
    are overridden.
    """
    # Load the base stylesheet (light theme). This is always present.
    try:
        base_css = CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{base_css}</style>", unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Failed to load base stylesheet: {exc}")
        return

    # If dark mode is active, load the dark overrides.
    if st.session_state.get("dark_mode", False):
        dark_path = Path(__file__).with_name("dark_style.css")
        try:
            dark_css = dark_path.read_text(encoding="utf-8")
            st.markdown(f"<style>{dark_css}</style>", unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Failed to load dark stylesheet: {exc}")



@st.cache_data(show_spinner="Loading NIRF data…")
def load_data() -> pd.DataFrame:
    return load_master()


def page_header(title: str, intro: str) -> None:
    st.title(title)
    st.markdown(f"<div class='nirf-intro'>{intro}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
def sidebar_filters(df: pd.DataFrame) -> dict:
    if "_pending_category" in st.session_state:
        st.session_state["g_category"] = st.session_state.pop("_pending_category")
    sb = st.sidebar
    sb.markdown("### Filters")
    cats = category_options(df)
    cat = sb.selectbox("Ranking category", cats, index=cats.index("Overall"), key="g_category",
                       help="Rankings are never mixed across categories: an institution can appear in several.")
    years = sorted(int(y) for y in df.loc[df["category"].eq(cat), "year"].unique())
    year = sb.select_slider("Year (single-year charts)", options=years, value=years[-1], key=f"g_year_{cat}")
    if len(years) > 1:
        rng = sb.select_slider("Year range (trend charts)", options=years, value=(years[0], years[-1]), key=f"g_range_{cat}")
    else:
        rng = (years[0], years[0])
    # Dark mode toggle

    exact = sb.toggle("Exact published ranks only", value=True, key="g_exact", help="Off: rank bands such as 101–150 are included using their midpoint as an estimate.")
    cov = coverage_summary(filter_data(df, category=cat))
    sb.caption(f"{cov['institutions']:,} institutions · {cov['exact_rank_rows']:,} exact ranks · {cov['band_rows']:,} banded · {cov['score_rows']:,} scored rows in {cat}.")
    return dict(category=cat, year=year, year_from=rng[0], year_to=rng[1], exact_only=exact)


def get_filters(df: pd.DataFrame) -> dict:
    return st.session_state.get("flt") or dict(category="Overall", year=year_range(df)[1], year_from=year_range(df)[0],
                                                year_to=year_range(df)[1], exact_only=True)


# ------------------------------------------------------------------ widgets for question-specific parameters
def _default_institution(df: pd.DataFrame, cat: str, nth: int = 0) -> str | None:
    d = filter_data(df, category=cat, exact_only=True)
    if d.empty:
        d = filter_data(df, category=cat)
    if d.empty:
        return None
    latest = d[d["year"].eq(d["year"].max())].sort_values("rank_value")
    return latest["institution_key"].iloc[min(nth, len(latest) - 1)]


def _widget(df: pd.DataFrame, qid: str, p: str, cat: str, defaults: dict, col):
    spec, key = PARAM_SPECS[p], f"w_{qid}_{p}_{cat if p in ('institution', 'institution_b') else ''}"
    kind, label = spec["kind"], spec["label"]
    default = defaults.get(p, spec.get("default"))
    with col:
        if kind == "institution":
            opts = institution_options(df, cat)
            keys = opts["institution_key"].tolist()
            names = dict(zip(opts["institution_key"], opts["institute_name"]))
            d = _default_institution(df, cat, 1 if p == "institution_b" else 0)
            return st.selectbox(label, keys, index=keys.index(d) if d in keys else 0, format_func=names.get, key=key) if keys else None
        if kind == "state":
            opts = state_options(df)
            if spec.get("optional"):
                pick = st.selectbox(label, ["All states"] + opts, key=key)
                return None if pick == "All states" else pick
            return st.selectbox(label, opts, index=opts.index(default) if default in opts else 0, key=key)
        if kind == "year":
            yrs = sorted(int(y) for y in df["year"].unique())
            return st.selectbox(label, yrs, index=yrs.index(default) if default in yrs else len(yrs) - 1, key=key)
        if kind == "choice":
            ch = spec["choices"]
            fmt = (lambda v: f"{v} · {PARAM_NAMES[v]}" if v in PARAM_NAMES else {"score": "Overall score", "rank_value": "Rank (estimated)"}.get(v, v))
            return st.selectbox(label, ch, index=ch.index(default) if default in ch else 0, format_func=fmt, key=key)
        if kind == "int":
            return st.slider(label, spec["min"], spec["max"], int(default), key=key)
        if kind == "float":
            return st.slider(label, float(spec["min"]), float(spec["max"]), float(default), step=float(spec.get("step", 0.05)), key=key)
        if kind == "bool":
            return st.toggle(label, value=bool(default), key=key)
    return default


def render_speech_button(text_content: str, element_id: str) -> None:
    """Render a lightweight HTML5 Web Speech Synthesis audio briefing button."""
    clean_text = text_content.replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
    html_code = f"""
    <div style="margin: 4px 0 8px 0;">
      <button id="spk_btn_{element_id}" onclick="spkToggle_{element_id}()" style="
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white; border: none; border-radius: 18px;
        padding: 4px 13px; font-size: 12px; font-weight: 600;
        cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
        box-shadow: 0 1px 3px rgba(37,99,235,0.25);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      ">
        <span id="spk_icon_{element_id}">🔊</span> <span id="spk_lbl_{element_id}">Listen to Takeaways</span>
      </button>
    </div>
    <script>
      let isPlaying_{element_id} = false;
      function spkToggle_{element_id}() {{
        if (!('speechSynthesis' in window)) return;
        const icon = document.getElementById('spk_icon_{element_id}');
        const lbl = document.getElementById('spk_lbl_{element_id}');
        if (isPlaying_{element_id}) {{
          window.speechSynthesis.cancel();
          isPlaying_{element_id} = false;
          icon.textContent = '🔊';
          lbl.textContent = 'Listen to Takeaways';
        }} else {{
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance('{clean_text}');
          u.rate = 1.05;
          u.onend = function() {{
            isPlaying_{element_id} = false;
            icon.textContent = '🔊';
            lbl.textContent = 'Listen to Takeaways';
          }};
          u.onerror = function() {{
            isPlaying_{element_id} = false;
            icon.textContent = '🔊';
            lbl.textContent = 'Listen to Takeaways';
          }};
          window.speechSynthesis.speak(u);
          isPlaying_{element_id} = true;
          icon.textContent = '⏹️';
          lbl.textContent = 'Stop Audio';
        }}
      }}
    </script>
    """
    components.html(html_code, height=36)


def render_question(df: pd.DataFrame, qid: str, flt: dict) -> None:
    q = QUESTIONS[qid]
    st.markdown(f"### <span class='nirf-qid'>{qid}</span>&nbsp; {q.title}", unsafe_allow_html=True)
    params = {p: flt[p] for p in q.params if p in GLOBAL_PARAMS and p in flt}
    inline = [p for p in q.params if p not in GLOBAL_PARAMS]
    defaults = default_params(df, qid)
    cat = params.get("category", flt["category"])
    if inline:
        with st.container(border=True):
            for i in range(0, len(inline), 4):
                chunk = inline[i:i + 4]
                cols = st.columns(len(chunk))
                for p, c in zip(chunk, cols):
                    params[p] = _widget(df, qid, p, cat, defaults, c)
    try:
        res = run_question(df, qid, **{k: v for k, v in params.items() if v is not None})
    except QueryValidationError as e:
        st.warning(str(e))
        return
    bullets, kpis = generate_insights(qid, res.params, res.data)
    if kpis:
        k_cols = st.columns(min(len(kpis), 4))
        for kc, kp in zip(k_cols, kpis[:4]):
            kpi_card(kc, kp["label"], kp["value"], kp.get("delta"), kp.get("help"))
    if bullets:
        st.markdown(
            "<div class='chat-insights-card'><h5>📌 Key Takeaways</h5><ul>" +
            "".join(f"<li>{_md_bold(b)}</li>" for b in bullets) +
            "</ul></div>",
            unsafe_allow_html=True
        )
        audio_text = "Key findings: " + ". ".join(b.replace("**", "").replace("*", "") for b in bullets)
        render_speech_button(audio_text, f"q_{qid.replace('.', '_')}")
    st.plotly_chart(
        res.figure, width="stretch", key=f"fig_{qid}",
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {"format": "png", "filename": f"nirf_{qid.replace('.', '_')}", "height": 900, "width": 1400, "scale": 2.5}
        }
    )
    with st.expander(f"Data behind this chart ({len(res.data):,} rows)"):
        st.dataframe(res.data, hide_index=True, width="stretch")
        st.download_button("Download CSV", res.data.to_csv(index=False).encode(), file_name=f"nirf_{qid.replace('.', '_')}.csv", mime="text/csv", key=f"dl_{qid}")


def render_section(df: pd.DataFrame, section: int, intro: str, title: str | None = None) -> None:
    flt = get_filters(df)
    page_header(title or SECTION_TITLES[section], intro)
    qs = by_section(section)
    ids = [q.qid for q in qs]
    choice = st.radio("Analysis", ids, format_func=lambda i: f"{i}  {SHORT[i]}", horizontal=True, key=f"sel_{section}", label_visibility="collapsed")
    st.markdown(f"<div class='nirf-note'>{SECTION_NOTES[section]}</div>", unsafe_allow_html=True)
    render_question(df, choice, flt)


def kpi_card(col, label, value, delta=None, help=None) -> None:
    """Key-figure card. Unlike st.metric it wraps long values (full institution names) instead of clipping them."""
    import html
    v = str(value)
    cls = "kpi-value kpi-long" if len(v) > 16 else "kpi-value"
    d = f"<div class='kpi-delta'>{html.escape(str(delta))}</div>" if delta else ""
    tip = f" title='{html.escape(str(help), quote=True)}'" if help else ""
    col.markdown(f"<div class='kpi-card'{tip}><div class='kpi-label'>{html.escape(str(label))}</div><div class='{cls}'>{html.escape(v)}</div>{d}</div>", unsafe_allow_html=True)


def _md_bold(text: str) -> str:
    """The takeaway cards are raw HTML, so turn **bold** markdown into <b> (otherwise the asterisks show literally)."""
    import re
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(text))
