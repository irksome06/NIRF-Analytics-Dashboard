"""Institution explorer: profile one institution, compare several, or analyze cross-discipline performance."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.institutions import compare_institutions, institution_radar
from app import ui
from engine.data_loader import PARAM_NAMES, institution_options
from engine.helpers import scope
from engine.query_engine import run_question


def generate_dossier_html(pick: str, cat: str, year: int, h: pd.DataFrame, tbl: pd.DataFrame, also: pd.DataFrame) -> str:
    cur_row = h[h["year"] == year]
    cur_rank = str(int(cur_row["rank"].iloc[0])) if not cur_row.empty and pd.notna(cur_row["rank"].iloc[0]) else "N/A"
    cur_score = f"{cur_row['score'].iloc[0]:.2f}" if not cur_row.empty and pd.notna(cur_row["score"].iloc[0]) else "N/A"
    city = str(h["city"].iloc[-1]) if "city" in h.columns and not h.empty and pd.notna(h["city"].iloc[-1]) else "N/A"
    state = str(h["state"].iloc[-1]) if "state" in h.columns and not h.empty and pd.notna(h["state"].iloc[-1]) else "N/A"

    traj_rows = ""
    for _, r in h.iterrows():
        rk_val = int(r["rank"]) if pd.notna(r.get("rank")) else r.get("rank_band_label", "Banded")
        sc_val = f"{r['score']:.2f}" if pd.notna(r.get("score")) else "—"
        tlr_val = f"{r['TLR']:.1f}" if pd.notna(r.get("TLR")) else "—"
        rpc_val = f"{r['RPC']:.1f}" if pd.notna(r.get("RPC")) else "—"
        go_val = f"{r['GO']:.1f}" if pd.notna(r.get("GO")) else "—"
        oi_val = f"{r['OI']:.1f}" if pd.notna(r.get("OI")) else "—"
        pr_val = f"{r['PR']:.1f}" if pd.notna(r.get("PR")) else "—"
        traj_rows += f"<tr><td><b>{int(r['year'])}</b></td><td>#{rk_val}</td><td><b>{sc_val}</b></td><td>{tlr_val}</td><td>{rpc_val}</td><td>{go_val}</td><td>{oi_val}</td><td>{pr_val}</td></tr>"

    diag_rows = ""
    if not tbl.empty and "parameter" in tbl.columns:
        for _, row in tbl.iterrows():
            diff = row["institution"] - row["cohort_median"]
            diff_color = "#10B981" if diff >= 0 else "#EF4444"
            diag_rows += f"<tr><td><b>{row['parameter']}</b></td><td>{PARAM_NAMES.get(row['parameter'], '')}</td><td><b>{row['institution']:.1f}</b></td><td>{row['cohort_median']:.1f}</td><td>{row['top10_average']:.1f}</td><td style='color:{diff_color};font-weight:600;'>{diff:+.1f}</td></tr>"

    cross_list = ""
    if not also.empty:
        for _, r in also.iterrows():
            rk = int(r["rank"]) if pd.notna(r.get("rank")) else r.get("rank_band_label", "Banded")
            sc = f"({r['score']:.1f} pts)" if pd.notna(r.get("score")) else ""
            cross_list += f"<span style='background:#EFF6FF;color:#1E40AF;padding:4px 10px;border-radius:6px;font-size:12px;margin-right:6px;'><b>{r['category']}:</b> #{rk} {sc}</span>"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NIRF Performance Dossier - {pick}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #0F172A; background: #F8FAFC; margin: 0; padding: 24px; }}
  .dossier {{ max-width: 900px; margin: 0 auto; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 32px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
  .header {{ border-bottom: 2px solid #2563EB; padding-bottom: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-start; }}
  h1 {{ margin: 0 0 6px 0; font-size: 24px; color: #1E293B; }}
  .sub {{ color: #64748B; font-size: 14px; margin-bottom: 8px; }}
  .badge-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  .badge-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-top: 3px solid #2563EB; border-radius: 8px; padding: 12px; text-align: center; }}
  .badge-label {{ font-size: 11px; font-weight: 600; color: #64748B; text-transform: uppercase; margin-bottom: 4px; }}
  .badge-val {{ font-size: 20px; font-weight: 700; color: #0F172A; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0 24px 0; font-size: 13px; }}
  th {{ background: #F1F5F9; color: #475569; text-align: left; padding: 10px 12px; font-weight: 600; border-bottom: 1px solid #CBD5E1; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #E2E8F0; }}
  tr:nth-child(even) {{ background: #FAFAFA; }}
  h2 {{ font-size: 16px; margin: 24px 0 8px 0; color: #1E40AF; border-left: 4px solid #2563EB; padding-left: 8px; }}
  .footer {{ font-size: 11px; color: #94A3B8; text-align: center; margin-top: 32px; border-top: 1px solid #E2E8F0; padding-top: 12px; }}
  @media print {{ body {{ background: white; padding: 0; }} .dossier {{ border: none; box-shadow: none; padding: 0; }} }}
</style>
</head>
<body>
<div class="dossier">
  <div class="header">
    <div>
      <div style="font-size:11px;font-weight:700;color:#2563EB;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">National Institutional Ranking Framework</div>
      <h1>{pick}</h1>
      <div class="sub">{city}, {state} · <b>Discipline:</b> {cat} · <b>Evaluation Edition:</b> {year}</div>
      <div style="margin-top:8px;">{cross_list}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:28px;font-weight:800;color:#2563EB;">#{cur_rank}</div>
      <div style="font-size:12px;font-weight:600;color:#64748B;">National Rank ({year})</div>
    </div>
  </div>

  <div class="badge-grid">
    <div class="badge-card"><div class="badge-label">Published Rank</div><div class="badge-val">#{cur_rank}</div></div>
    <div class="badge-card"><div class="badge-label">Overall Score</div><div class="badge-val">{cur_score}</div></div>
    <div class="badge-card"><div class="badge-label">Best Rank (2017–25)</div><div class="badge-val">#{int(h['rank'].min()) if not h.empty and pd.notna(h['rank'].min()) else '—'}</div></div>
    <div class="badge-card"><div class="badge-label">Editions Listed</div><div class="badge-val">{h['year'].nunique()} / 9</div></div>
  </div>

  <h2>Parameter Performance Diagnostic ({year})</h2>
  <table>
    <thead><tr><th>Code</th><th>Parameter Name</th><th>Score</th><th>Cohort Median</th><th>Top-10 Average</th><th>Δ vs Median</th></tr></thead>
    <tbody>{diag_rows}</tbody>
  </table>

  <h2>Historical Trajectory Across Editions (2017–2025)</h2>
  <table>
    <thead><tr><th>Year</th><th>Rank</th><th>Score</th><th>TLR</th><th>RPC</th><th>GO</th><th>OI</th><th>PR</th></tr></thead>
    <tbody>{traj_rows}</tbody>
  </table>

  <div class="footer">
    Official NIRF Institutional Dossier · Data Source: Ministry of Education, Government of India (NIRF 2017–2025). Generated via NIRF Analytics Dashboard.
  </div>
</div>
</body>
</html>"""


def page() -> None:
    df = ui.load_data()
    flt = ui.get_filters(df)
    cat, year = flt["category"], flt["year"]
    ui.page_header("Institution explorer", f"Profile one institution, compare several, or evaluate cross-discipline performance across India.")
    names = institution_options(df, cat)["institute_name"].tolist()
    if not names:
        st.info("No institutions in this category.")
        return

    tab1, tab2, tab3 = st.tabs(["Profile", "Compare", "Cross-Discipline Matrix"])

    with tab1:
        default = names.index("Indian Institute of Technology Madras") if "Indian Institute of Technology Madras" in names else 0
        pick = st.selectbox("Search an institution", names, index=default, key="inst_pick")
        h = scope(df, cat, year_from=flt["year_from"], year_to=flt["year_to"])
        h = h[h["institute_name"].eq(pick)].sort_values("year")
        ex = h[h["rank_status"].eq("exact")]
        c = st.columns(4)
        c[0].metric("Latest exact rank", f"{int(ex['rank'].iloc[-1])}" if len(ex) else "n/a", help=f"Edition {int(ex['year'].iloc[-1])}" if len(ex) else None)
        c[1].metric("Best exact rank", f"{int(ex['rank'].min())}" if len(ex) else "n/a")
        c[2].metric("Editions listed", f"{h['year'].nunique()}")
        c[3].metric("State", h["state"].iloc[-1] if len(h) else "n/a")

        # Multi-category appearances in selected year
        also = df[df["institute_name"].eq(pick) & df["year"].eq(year) & df["rank_status"].isin(["exact", "band"])].sort_values("rank_value")
        if len(also) > 1:
            pills = " ".join(f"<span class='chat-badge chat-badge-rules'><b>{r['category']}:</b> #{int(r['rank']) if pd.notna(r['rank']) else r['rank_band_label']} ({r['score']:.1f} pts)</span>" if pd.notna(r.get('score')) else f"<span class='chat-badge chat-badge-meta'><b>{r['category']}:</b> #{int(r['rank']) if pd.notna(r['rank']) else r['rank_band_label']}</span>" for _, r in also.iterrows())
            st.markdown(f"<div style='margin:.5rem 0 .9rem 0'><small style='color:var(--muted);font-weight:600;margin-right:.5rem;'>All categories in {year}:</small>{pills}</div>", unsafe_allow_html=True)

        left, right = st.columns(2, gap="large")
        with left:
            res = run_question(df, "1.1", category=cat, institution=pick, year_from=flt["year_from"], year_to=flt["year_to"])
            st.plotly_chart(res.figure, width="stretch", key="inst_path", config={"displaylogo": False})
        with right:
            tbl, fig = institution_radar(df, cat, pick, year)
            st.plotly_chart(fig, width="stretch", key="inst_radar", config={"displaylogo": False})
        
        # 5-Parameter KPI breakdown against cohort median
        if not tbl.empty and "cohort_median" in tbl.columns and "institution" in tbl.columns:
            st.markdown(f"##### {year} Parameter Breakdown vs Cohort Median")
            p_cols = st.columns(5)
            for k, (_, row) in enumerate(tbl.iterrows()):
                p_code = row["parameter"]
                inst_score = row["institution"]
                med_score = row["cohort_median"]
                diff = inst_score - med_score
                p_cols[k].metric(label=p_code, value=f"{inst_score:.1f}", delta=f"{diff:+.1f} vs median")

        with st.expander("Full History Table"):
            st.dataframe(h[["year", "rank", "rank_status", "rank_band_label", "score", "TLR", "RPC", "GO", "OI", "PR"]], hide_index=True, width="stretch")

        # Official Report Card Dossier Download Button
        dossier_html = generate_dossier_html(pick, cat, year, h, tbl, also)
        st.download_button(
            label="📄 Download Official Institutional Report Card (HTML / Print to PDF)",
            data=dossier_html.encode("utf-8"),
            file_name=f"nirf_report_card_{pick.replace(' ', '_')}_{year}.html",
            mime="text/html",
            key="dl_dossier"
        )

    with tab2:
        defaults = [n for n in names if n.startswith("Indian Institute of Technology")][:3]
        picks = st.multiselect("Choose up to 5 institutions", names, default=defaults, max_selections=5, key="inst_cmp")
        if picks:
            tbl, fig = compare_institutions(df, cat, picks, flt["year_from"], flt["year_to"], flt["exact_only"])
            st.plotly_chart(fig, width="stretch", key="inst_cmp_fig", config={"displaylogo": False})
            with st.expander("Data table"):
                st.dataframe(tbl, hide_index=True, width="stretch")
        else:
            st.info("Pick at least one institution.")

    with tab3:
        st.markdown(f"### Multi-Discipline National Footprint Matrix ({year})")
        st.caption("How premier Indian institutions perform across all official disciplines simultaneously:")
        # Find top 20 institutions from Overall or Engineering
        top_cohort = df[df["category"].isin(["Overall", "Engineering"]) & df["year"].eq(year) & df["rank_status"].eq("exact")].sort_values("rank")["institute_name"].unique()[:20]
        matrix_df = df[df["institute_name"].isin(top_cohort) & df["year"].eq(year) & df["rank_status"].isin(["exact", "band"])].copy()
        matrix_df["rank_display"] = matrix_df.apply(lambda r: f"#{int(r['rank'])}" if pd.notna(r['rank']) else str(r['rank_band_label']), axis=1)
        pivot_matrix = matrix_df.pivot_table(index="institute_name", columns="category", values="rank_display", aggfunc="first").fillna("—")
        st.dataframe(pivot_matrix, width="stretch")
