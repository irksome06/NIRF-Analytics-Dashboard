"""Shared helpers for every analysis module: theme, filtering shortcuts, rank groups, small stats."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from engine.data_loader import PARAMS, filter_data

# ---------------------------------------------------------------- theme
# Palette: ink-indigo as the brand colour, marigold for emphasis, vermilion for decline.
C = dict(ink="#1A1D3A", brand="#343FA0", brand_light="#A9B0E3", amber="#E3A008", gold="#F59E0B", coral="#D4483B",
         slate="#5E6278", mist="#E3E4EC", green="#1E8B6A", violet="#6D4FC2")
SERIES = ["#343FA0", "#E3A008", "#D4483B", "#178A8F", "#8E4585", "#4C9A5F", "#1A1D3A", "#9A9FB5"]
PARAM_COLORS = dict(zip(PARAMS, ["#343FA0", "#E3A008", "#D4483B", "#178A8F", "#8E4585"]))
FONT = "Public Sans, Segoe UI, Helvetica, Arial, sans-serif"
DISPLAY_FONT = "Bricolage Grotesque, Public Sans, Segoe UI, Helvetica, Arial, sans-serif"
ESTIMATE_NOTE = "Hollow / light marks are rank-band midpoint estimates, not published exact ranks."


def style(fig: go.Figure, title: str, subtitle: str | None = None, height: int = 460,
          legend_title: str | None = None) -> go.Figure:
    text = f"<b>{title}</b>" + (f"<br><span style='font-size:12px;font-weight:400;color:{C['slate']}'>{subtitle}</span>" if subtitle else "")
    fig.update_layout(
        template="plotly_white", colorway=SERIES, height=height,
        title=dict(text=text, x=0.0, xanchor="left", font=dict(family=DISPLAY_FONT, size=17, color=C["ink"])),
        font=dict(family=FONT, size=12, color=C["ink"]),
        margin=dict(l=60, r=30, t=92 if subtitle else 70, b=60),
        legend=dict(title=legend_title, orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, font=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family=FONT, bgcolor=C["ink"], font_color="white", bordercolor=C["ink"]),
    )
    fig.update_xaxes(gridcolor="#ECEDF4", zerolinecolor="#D9DBE6", linecolor="#D9DBE6", tickfont=dict(color=C["slate"]),
                     title_font=dict(color=C["slate"], size=12))
    fig.update_yaxes(gridcolor="#ECEDF4", zerolinecolor="#D9DBE6", showline=False, tickfont=dict(color=C["slate"]),
                     title_font=dict(color=C["slate"], size=12))
    return fig


def empty_fig(message: str, title: str = "No data for these filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=C["slate"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, title, height=320)


# ---------------------------------------------------------------- data shortcuts
def scope(df: pd.DataFrame, category: str | None = None, year=None, year_from=None, year_to=None,
          exact_only: bool = False, require_score: bool = False, states=None) -> pd.DataFrame:
    years = year
    if years is None and year_from is not None and year_to is not None:
        years = (int(year_from), int(year_to))
    return filter_data(df, years=years, category=category, exact_only=exact_only,
                       require_score=require_score, states=states)


def rank_col(exact_only: bool) -> str:
    return "rank" if exact_only else "rank_value"


def rank_label(d: pd.DataFrame) -> pd.Series:
    """Human label for a rank: '12' for exact, '101-150 (band)' for bands."""
    exact = d["rank"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    band = d["rank_band_label"].fillna("").astype(str) + " (band)"
    return np.where(d["rank_status"].eq("exact"), exact, np.where(d["rank_status"].eq("band"), band, "not ranked"))


GROUP_ORDER = ["Top 10", "11–50", "51–100", "101–200", "201+", "Not ranked"]


def rank_group(d: pd.DataFrame) -> pd.Series:
    """Rank group built from rank_low/rank_high so exact ranks and published bands share one scale."""
    low, high = d["rank_low"], d["rank_high"]
    out = np.select(
        [high <= 10, (low >= 11) & (high <= 50), (low >= 51) & (high <= 100),
         (low >= 101) & (high <= 200), low >= 201],
        ["Top 10", "11–50", "51–100", "101–200", "201+"], default="Not ranked")
    return pd.Series(out, index=d.index)


def pair_ranks(df: pd.DataFrame, category: str, year_a: int, year_b: int, exact_only: bool = True) -> pd.DataFrame:
    """Institutions ranked in both years. rank_change > 0 = moved up (rank number fell)."""
    col = rank_col(exact_only)
    keep = ["institution_key", "institute_name", "state", col, "rank_status", "rank_low", "rank_high"]
    a = scope(df, category, year_a, exact_only=exact_only)[keep].dropna(subset=[col])
    b = scope(df, category, year_b, exact_only=exact_only)[keep].dropna(subset=[col])
    m = a.merge(b, on="institution_key", suffixes=("_a", "_b"))
    m = m.rename(columns={"institute_name_a": "institute_name", "state_a": "state",
                          f"{col}_a": "rank_a", f"{col}_b": "rank_b",
                          "rank_status_a": "status_a", "rank_status_b": "status_b"})
    m["rank_change"] = m["rank_a"] - m["rank_b"]
    m["basis"] = np.where((m["status_a"] == "exact") & (m["status_b"] == "exact"), "Exact ranks", "Includes band estimate")
    return m[["institution_key", "institute_name", "state", "rank_a", "rank_b", "rank_change",
              "status_a", "status_b", "basis"]]


# ---------------------------------------------------------------- small stats
def ols(x: pd.Series, y: pd.Series):
    """Slope, intercept, Pearson r, Spearman rho, n  (no statsmodels dependency)."""
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    if n < 3 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return None
    xv, yv = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
    slope, intercept = np.polyfit(xv, yv, 1)
    r = float(np.corrcoef(xv, yv)[0, 1])
    rho = float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))
    return dict(slope=float(slope), intercept=float(intercept), r=r, rho=rho, n=n)


def robust_z(s: pd.Series) -> pd.Series:
    """Median/MAD z-score; falls back to ordinary z when MAD is 0."""
    med = s.median()
    mad = (s - med).abs().median()
    if mad and mad > 0:
        return 0.6745 * (s - med) / mad
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def iqr(s: pd.Series) -> float:
    return float(s.quantile(0.75) - s.quantile(0.25))


# ---------------------------------------------------------------- institution names for charts
_ABBR = [
    ("Indian Institute of Science Education and Research", "IISER"), ("Indian Institute of Information Technology", "IIIT"),
    ("International Institute of Information Technology", "IIIT"), ("Indian Institute of Technology", "IIT"),
    ("Indian Institute of Management", "IIM"), ("National Institute of Technology", "NIT"),
    ("All India Institute of Medical Sciences", "AIIMS"), ("Birla Institute of Technology & Science", "BITS"),
    ("Birla Institute of Technology and Science", "BITS"), ("Vellore Institute of Technology", "VIT"),
]


def short_name(name: str, maxlen: int | None = None, abbr: bool = True) -> str:
    """Readable short form of an institution name. Abbreviates well-known families ('Indian Institute of Technology Madras' ->
    'IIT Madras') and, if still longer than maxlen, cuts at a word boundary with an ellipsis. Never cuts mid-word."""
    n = str(name).strip()
    if abbr:
        for full, short in _ABBR:
            if n.startswith(full):
                n = (short + n[len(full):]).replace(" -", " ").replace("  ", " ").strip(" ,-")
                n = n.replace("(Banaras Hindu University)", "(BHU)").replace("(Indian School of Mines)", "(ISM)")
                break
        if n == "Indian Institute of Science":
            n = "IISc"
    if maxlen and len(n) > maxlen:
        head = n[: maxlen - 1]
        cut = head.rsplit(" ", 1)[0].rstrip(" ,-(") if " " in head and n[maxlen - 1] != " " else head.rstrip(" ,-(")
        n = (cut or head) + "…"
    return n


def short_labels(names, maxlen: int = 28) -> list[str]:
    return [short_name(n, maxlen) for n in names]


def stagger(n: int) -> list[str]:
    """Rotate label positions so neighbouring points' labels overlap less."""
    pos = ["top right", "bottom left", "top left", "bottom right"]
    return [pos[i % 4] for i in range(n)]


def _wrap(name: str, width: int = 38) -> list[str]:
    """Wrap a name onto several lines at word boundaries (never cuts a word)."""
    lines, cur = [], ""
    for w in str(name).split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur] if cur else [""]


def name_labels(fig: go.Figure, xs, ys, names, all_x, all_y, reverse_y: bool = False, color: str | None = None) -> go.Figure:
    """Label highlighted points with their FULL institution names. Names are stacked in a column in the right margin and joined to
    their points by thin leader lines, so nothing is clipped and labels cannot sit on top of each other. Call after style()."""
    xs, ys, names = list(xs), list(ys), list(names)
    if not xs:
        return fig
    color = color or C["coral"]
    ax_vals, ay_vals = np.asarray(all_x, float), np.asarray(all_y, float)
    px_, py_ = (ax_vals.max() - ax_vals.min()) or 1.0, (ay_vals.max() - ay_vals.min()) or 1.0
    x0, x1 = ax_vals.min() - 0.04 * px_, ax_vals.max() + 0.04 * px_
    y_lo, y_hi = ay_vals.min() - 0.05 * py_, ay_vals.max() + 0.05 * py_
    top_val, bot_val = (y_lo, y_hi) if reverse_y else (y_hi, y_lo)  # data value at the top / bottom edge of the plot
    wrapped = [_wrap(n) for n in names]
    heights = [13 * len(w) + 5 for w in wrapped]
    lay = fig.layout
    margin_t, margin_b = lay.margin.t or 90, lay.margin.b or 60
    plot_px = max(120, (lay.height or 520) - margin_t - margin_b)
    need = sum(heights)
    if need > plot_px:  # many labels: make the chart taller instead of shrinking or dropping names
        plot_px = need + 10
        fig.update_layout(height=int(plot_px + margin_t + margin_b))
    span = bot_val - top_val
    pix = [(y - top_val) / span * plot_px for y in ys]  # pixel row of each point, 0 = top
    order = sorted(range(len(xs)), key=lambda i: pix[i])
    pos = {}
    prev_end = 0.0
    for i in order:  # push labels down so none overlap
        start = max(pix[i] - heights[i] / 2, prev_end)
        pos[i] = start
        prev_end = start + heights[i]
    nxt = plot_px
    for i in reversed(order):  # if that ran past the bottom, pull back up
        pos[i] = min(pos[i], nxt - heights[i])
        nxt = pos[i]
    label_x = x1 + 0.02 * (x1 - x0)
    for i in range(len(xs)):
        centre = pos[i] + heights[i] / 2
        fig.add_annotation(x=xs[i], y=ys[i], xref="x", yref="y", ax=label_x, ay=top_val + centre / plot_px * span, axref="x", ayref="y",
                           text="<br>".join(wrapped[i]), showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="rgba(94,98,120,0.55)", opacity=1,
                           xanchor="left", yanchor="middle", align="left", font=dict(size=11, color=C["ink"]), standoff=4, startstandoff=2)
    longest = max(len(l) for w in wrapped for l in w)
    fig.update_layout(margin=dict(r=int(longest * 6.3 + 28)))
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[bot_val, top_val])
    return fig
