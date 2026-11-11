"""Intelligent insight generator and KPI summarizer for NIRF queries.
Generates natural-language executive summaries and key metric cards from query results.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def generate_insights(qid: str, params: dict, data: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """
    Given question ID, parameters, and result DataFrame,
    returns (list_of_bullet_insights, list_of_kpi_metrics).
    """
    if data is None or data.empty:
        return ["No records match the selected criteria."], []

    bullets: list[str] = []
    metrics: list[dict] = []

    try:
        # 1.1 Institution Rank Trend
        if qid == "1.1":
            if "year" in data.columns and ("score" in data.columns or "rank_label" in data.columns):
                sorted_d = data.sort_values("year")
                inst = params.get("institution", "Institution")
                first_row = sorted_d.iloc[0]
                last_row = sorted_d.iloc[-1]
                
                cur_rank = str(last_row.get("rank_label", "N/A"))
                start_rank = str(first_row.get("rank_label", "N/A"))
                cur_score = last_row.get("score")
                
                metrics.append({"label": "Current Rank", "value": f"#{cur_rank}" if cur_rank.isdigit() else cur_rank})
                if pd.notna(cur_score):
                    metrics.append({"label": "Latest Score", "value": f"{cur_score:.2f}"})
                metrics.append({"label": "Years Ranked", "value": f"{len(sorted_d)} / 9"})

                if cur_rank.isdigit() and start_rank.isdigit():
                    diff = int(start_rank) - int(cur_rank)
                    if diff > 0:
                        change_str = f"climbed {diff} positions"
                        metrics.append({"label": "Overall Shift", "value": f"+{diff} ranks", "delta": f"From #{start_rank}"})
                    elif diff < 0:
                        change_str = f"slipped {abs(diff)} positions"
                        metrics.append({"label": "Overall Shift", "value": f"-{abs(diff)} ranks", "delta": f"From #{start_rank}"})
                    else:
                        change_str = "retained its exact rank"
                        metrics.append({"label": "Overall Shift", "value": "Stable", "delta": f"Rank #{cur_rank}"})
                    bullets.append(f"Rank trajectory: Started at #{start_rank} in {first_row['year']} and {change_str} to #{cur_rank} in {last_row['year']}.")
                else:
                    bullets.append(f"Ranked at {start_rank} in {first_row['year']} and {cur_rank} in {last_row['year']}.")

                if "score" in sorted_d.columns and sorted_d["score"].notna().any():
                    best_score_row = sorted_d.loc[sorted_d["score"].idxmax()]
                    bullets.append(f"Peak score achieved was **{best_score_row['score']:.2f}** in {int(best_score_row['year'])}.")

        # 1.2 Biggest Improvers
        elif qid in ("1.2", "7.4"):
            if "rank_a" in data.columns and "rank_b" in data.columns and "institute_name" in data.columns:
                top_improver = data.iloc[0]
                gain = int(top_improver["rank_a"]) - int(top_improver["rank_b"])
                metrics.append({"label": "Top Climber", "value": f"+{gain} ranks"})
                metrics.append({"label": "Leading Institution", "value": str(top_improver['institute_name'])})
                bullets.append(f"**{top_improver['institute_name']}** recorded the largest upward jump, leaping **+{gain} positions** (from #{int(top_improver['rank_a'])} to #{int(top_improver['rank_b'])}).")
                if len(data) > 1:
                    runner_up = data.iloc[1]
                    gain_2 = int(runner_up["rank_a"]) - int(runner_up["rank_b"])
                    bullets.append(f"Runner-up climber: **{runner_up['institute_name']}** (+{gain_2} positions, #{int(runner_up['rank_a'])} → #{int(runner_up['rank_b'])}).")

        # 1.3 Biggest Decliners
        elif qid == "1.3":
            if "rank_a" in data.columns and "rank_b" in data.columns and "institute_name" in data.columns:
                top_decliner = data.iloc[0]
                drop = int(top_decliner["rank_b"]) - int(top_decliner["rank_a"])
                metrics.append({"label": "Largest Drop", "value": f"-{drop} ranks"})
                metrics.append({"label": "Institution", "value": str(top_decliner['institute_name'])})
                bullets.append(f"**{top_decliner['institute_name']}** experienced the largest decline, dropping **-{drop} positions** (from #{int(top_decliner['rank_a'])} down to #{int(top_decliner['rank_b'])}).")

        # 1.4 Average Score Over Time
        elif qid == "1.4":
            if "mean" in data.columns and "year" in data.columns:
                yr_min, yr_max = data["year"].min(), data["year"].max()
                mean_first = data[data["year"] == yr_min]["mean"].mean()
                mean_last = data[data["year"] == yr_max]["mean"].mean()
                delta = mean_last - mean_first
                metrics.append({"label": f"Mean Score ({yr_max})", "value": f"{mean_last:.2f}"})
                metrics.append({"label": "9-Year Change", "value": f"{delta:+.2f} pts"})
                bullets.append(f"Average score across ranked institutions changed by **{delta:+.2f} points** from {yr_min} ({mean_first:.2f}) to {yr_max} ({mean_last:.2f}).")

        # 1.5 Consistent Top 10
        elif qid == "1.5":
            if "institute_name" in data.columns and "years_in_top10" in data.columns:
                perfect = data[data["years_in_top10"] == data["years_in_top10"].max()]
                max_yrs = int(data["years_in_top10"].max())
                metrics.append({"label": "Most Consistent", "value": f"{len(perfect)} colleges"})
                metrics.append({"label": "Top-10 Tenures", "value": f"{max_yrs} editions"})
                names = ", ".join(f"**{n}**" for n in perfect["institute_name"].head(3))
                bullets.append(f"{len(perfect)} institution(s) remained in the top 10 for {max_yrs} editions, led by {names}.")

        # 1.6 Top Institutions / Leaderboard
        elif qid == "1.6":
            if "institute_name" in data.columns and "score" in data.columns:
                top_row = data.iloc[0]
                metrics.append({"label": "National Rank 1", "value": str(top_row['institute_name'])})
                metrics.append({"label": "Top Score", "value": f"{top_row['score']:.2f}"})
                if len(data) >= 3:
                    top3 = [f"**#{int(r.get('position', i+1))} {r['institute_name']}** ({r['score']:.1f})" for i, (_, r) in enumerate(data.head(3).iterrows())]
                    bullets.append("Top 3 institutions: " + " · ".join(top3))
                if "state" in data.columns:
                    top_state = data["state"].mode()[0]
                    bullets.append(f"**{top_state}** has the highest representation in this leaderboard ({data['state'].eq(top_state).sum()} institutions).")

        # 1.7 Compare Two Institutions
        elif qid == "1.7":
            if "institute_name" in data.columns and "score" in data.columns:
                latest_yr = data["year"].max() if "year" in data.columns else None
                sub = data[data["year"] == latest_yr] if latest_yr else data
                if len(sub) >= 2:
                    r1, r2 = sub.iloc[0], sub.iloc[1]
                    margin = abs(r1["score"] - r2["score"])
                    lead = r1 if r1["score"] >= r2["score"] else r2
                    trail = r2 if lead is r1 else r1
                    metrics.append({"label": "Current Leader", "value": str(lead['institute_name'])})
                    metrics.append({"label": "Score Margin", "value": f"+{margin:.2f} pts"})
                    bullets.append(f"In {latest_yr or 'the latest edition'}, **{lead['institute_name']}** leads **{trail['institute_name']}** by **{margin:.2f} score points**.")

        # 2.1 Top States in Top 100
        elif qid == "2.1":
            if "state" in data.columns and "institutions" in data.columns:
                leader = data.iloc[0]
                share = float(leader["share_pct"]) if "share_pct" in data.columns else (leader["institutions"] / data["institutions"].sum() * 100)
                metrics.append({"label": "Top State", "value": str(leader['state'])})
                metrics.append({"label": "Top-100 Share", "value": f"{leader['institutions']} ({share:.0f}%)"})
                bullets.append(f"**{leader['state']}** leads the country with **{leader['institutions']} institutions** in the top 100 ({share:.1f}% national share).")
                if len(data) >= 3:
                    r2, r3 = data.iloc[1], data.iloc[2]
                    bullets.append(f"Followed by **{r2['state']}** ({r2['institutions']}) and **{r3['state']}** ({r3['institutions']}).")

        # 2.4 State vs State
        elif qid == "2.4":
            if "state" in data.columns and "top100_count" in data.columns:
                st_grp = data.groupby("state")["top100_count"].mean()
                if len(st_grp) >= 2:
                    s1, s2 = st_grp.index[0], st_grp.index[1]
                    c1, c2 = st_grp.iloc[0], st_grp.iloc[1]
                    winner = s1 if c1 >= c2 else s2
                    loser = s2 if winner == s1 else s1
                    metrics.append({"label": f"{winner} (Avg Top-100)", "value": f"{max(c1, c2):.1f}"})
                    metrics.append({"label": f"{loser} (Avg Top-100)", "value": f"{min(c1, c2):.1f}"})
                    bullets.append(f"**{winner}** averages **{max(c1, c2):.1f} top-100 institutions**, outpacing **{loser}** (averaging {min(c1, c2):.1f}).")

        # 3.1 & 3.3 Score Distributions & Summary
        elif qid in ("3.1", "3.3"):
            if "score" in data.columns and data["score"].notna().any():
                sc = data["score"].dropna()
                mean_val, med_val, min_val, max_val = sc.mean(), sc.median(), sc.min(), sc.max()
                metrics.append({"label": "Mean Score", "value": f"{mean_val:.2f}"})
                metrics.append({"label": "Median Score", "value": f"{med_val:.2f}"})
                metrics.append({"label": "Score Range", "value": f"{min_val:.1f} – {max_val:.1f}"})
                bullets.append(f"Overall distribution spans from **{min_val:.2f}** to **{max_val:.2f}** with a median of **{med_val:.2f}** (mean {mean_val:.2f}).")
            elif "statistic" in data.columns and "value" in data.columns:
                stats = dict(zip(data["statistic"], data["value"]))
                for k in ["mean", "median", "min", "max"]:
                    if k in stats:
                        metrics.append({"label": k.title(), "value": f"{float(stats[k]):.2f}"})
                bullets.append(f"Scores center around median **{float(stats.get('median', 0)):.2f}** (mean {float(stats.get('mean', 0)):.2f}), ranging from {float(stats.get('min', 0)):.2f} to {float(stats.get('max', 0)):.2f}.")

        # 4.1, 4.2, 4.3, 4.4, 4.6 Correlations
        elif qid in ("4.1", "4.2", "4.3", "4.4", "4.6"):
            if qid == "4.3" and "pearson_r" in data.columns:
                strongest = data.sort_values("pearson_r", ascending=False).iloc[0]
                metrics.append({"label": "Strongest Driver", "value": str(strongest['parameter'])})
                metrics.append({"label": "Pearson r", "value": f"{strongest['pearson_r']:.3f}"})
                bullets.append(f"**{strongest['parameter']} ({strongest.get('name', '')})** has the highest correlation with overall score (**r = {strongest['pearson_r']:.3f}**).")
            elif "score" in data.columns:
                # find parameter column
                param_col = next((c for c in ["RPC", "TLR", "GO", "OI", "PR"] if c in data.columns), None)
                if param_col:
                    r = data[[param_col, "score"]].dropna().corr().iloc[0, 1]
                    metrics.append({"label": f"Correlation (r)", "value": f"{r:.3f}"})
                    metrics.append({"label": "Variance (R²)", "value": f"{r**2*100:.1f}%"})
                    strength = "exceptionally strong" if abs(r) > 0.8 else "moderate" if abs(r) > 0.5 else "weak"
                    bullets.append(f"**{param_col}** demonstrates a **{strength} correlation** with overall score (**r = {r:.3f}**, explaining {r**2*100:.1f}% of variance).")

        # 5.1 Top 5 Radar
        elif qid == "5.1":
            if "institute_name" in data.columns and "score" in data.columns:
                leader = data.iloc[0]
                metrics.append({"label": "National Rank 1", "value": str(leader['institute_name'])})
                metrics.append({"label": "Leader Score", "value": f"{leader['score']:.2f}"})
                bullets.append(f"**{leader['institute_name']}** leads the group across composite metrics with an aggregate score of **{leader['score']:.2f}**.")

        # 6.1 Top 10 vs Rest
        elif qid == "6.1":
            if "group" in data.columns and "mean" in data.columns:
                top10_m = data[data["group"].str.contains("Top 10", case=False)]["mean"].iloc[0]
                rest_m = data[~data["group"].str.contains("Top 10", case=False)]["mean"].iloc[0]
                gap = top10_m - rest_m
                metrics.append({"label": "Top 10 Mean", "value": f"{top10_m:.2f}"})
                metrics.append({"label": "Remaining Mean", "value": f"{rest_m:.2f}"})
                metrics.append({"label": "Elite Margin", "value": f"+{gap:.2f} pts"})
                bullets.append(f"Top 10 institutions outperform the rest of the cohort by an average of **{gap:.2f} score points** ({top10_m:.1f} vs {rest_m:.1f}).")

        # 7.1 Promotion into Top 50
        elif qid == "7.1":
            promoted_count = len(data)
            metrics.append({"label": "Promoted to Top 50", "value": f"{promoted_count} colleges"})
            if promoted_count > 0 and "institute_name" in data.columns:
                names = ", ".join(f"**{n}**" for n in data["institute_name"].head(3))
                bullets.append(f"**{promoted_count} institutions** successfully jumped from the 51–100 band into the top 50, including {names}.")
            else:
                bullets.append("No institutions transitioned from the 51–100 band into the top 50 between these years.")

        # 8.1 & 8.3 Outliers
        elif qid in ("8.1", "8.3"):
            if qid == "8.1" and "institute_name" in data.columns:
                metrics.append({"label": "Outliers Identified", "value": f"{len(data)} institutions"})
                top_outlier = data.iloc[0]
                bullets.append(f"**{top_outlier['institute_name']}** exhibits the most disproportionate research output relative to its overall score.")
            elif qid == "8.3" and "change" in data.columns:
                top_chg = data.iloc[0]
                metrics.append({"label": "Largest Score Swing", "value": f"{top_chg['change']:+.2f} pts"})
                bullets.append(f"**{top_chg['institute_name']}** logged an anomalous score fluctuation of **{top_chg['change']:+.2f} points**.")

        # Custom Builder Query Fallback
        elif qid == "custom":
            metric_col = next((c for c in ["score", "TLR", "RPC", "GO", "OI", "PR", "rank_value", "institutions", "count"] if c in data.columns), None)
            if metric_col and pd.api.types.is_numeric_dtype(data[metric_col]):
                vals = data[metric_col].dropna()
                metrics.append({"label": f"Average {metric_col}", "value": f"{vals.mean():.2f}"})
                metrics.append({"label": f"Median {metric_col}", "value": f"{vals.median():.2f}"})
                metrics.append({"label": "Data Points", "value": f"{len(data):,}"})
                group_col = data.columns[0]
                top_item = data.sort_values(metric_col, ascending=False).iloc[0]
                bullets.append(f"Highest value observed in **{top_item[group_col]}** with {metric_col} of **{top_item[metric_col]:.2f}**.")

    except Exception:
        pass  # Graceful fallback: never crash if dataframe has unexpected custom shape

    return bullets, metrics
