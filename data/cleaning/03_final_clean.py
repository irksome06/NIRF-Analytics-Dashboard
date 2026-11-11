#!/usr/bin/env python3
"""STEP 3 - FINAL CLEANING: output/nirf_master.csv -> ../nirf_master_final.csv  (+ cleaning_log.csv)

    python 03_final_clean.py [src.csv] [out.csv]

Only fixes that are verifiable from the data itself. No imputation of ranks, scores or IDs."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent
# default chain: 02_preprocess.py -> output/nirf_master.csv  ->  this script  ->  ../nirf_master_final.csv (what the dashboard loads)
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "output" / "nirf_master.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE.parent / "nirf_master_final.csv"
LOG = BASE / "cleaning_log.csv"
df = pd.read_csv(SRC)
log = []
def note(step, n, detail): log.append((step, int(n), detail)); print(f"[{step}] {n} rows - {detail}")

# 1. Rank withdrawn mis-coded as exact (raw score field says 'Rank Withdrawn*') -------------
m = (df.category == "Dental") & (df.year == 2025) & (df.rank_status == "exact") & df.score.isna()
for c in ["rank", "rank_value", "rank_low", "rank_high", "rank_midpoint"]: df.loc[m, c] = np.nan
df.loc[m, "rank_status"] = "withdrawn"; df.loc[m, "rank_original"] = "Rank Withdrawn*"
df.loc[m, "weight_scheme"] = df.loc[m, "weight_scheme"]  # unchanged
note("withdrawn_fix", m.sum(), "Dental 2025 exact rank 18 -> withdrawn (raw score = 'Rank Withdrawn*')")

# 2. Name text glitches (canonical name only; *_original left as audit) ---------------------
def fix_name(n):
    n = n.replace("''", "'")
    n = re.sub(r"(?<=\w)\?(?=s\b)", "'", n)          # Mandal?s -> Mandal's
    n = re.sub(r"\s+\d{3}\s*$", "", n)               # stray trailing '511'
    return n.strip()
fixed = df.institute_name.map(fix_name)
note("name_text", (fixed != df.institute_name).sum(), "fixed '?' apostrophes, doubled apostrophes, trailing digits")
df["institute_name"] = fixed

# 3. Merge institutions split by punctuation/spacing variants --------------------------------
norm = df.institute_name.map(lambda n: re.sub(r"[^a-z0-9]", "", n.lower()))
OVERRIDE = {"gokarajurangarajuinstituteofengineeringtechnology":
            "Gokaraju Rangaraju Institute of Engineering & Technology"}
merged = 0
for nk, g in df.groupby(norm):
    keys = g.institution_key.unique()
    if len(keys) < 2: continue
    if g.duplicated(["year", "category"]).any():      # co-listed same year -> genuinely different (e.g. CMR KA vs TG)
        print("kept separate:", g.institute_name.unique().tolist()); continue
    names = g.institute_name.value_counts()
    best = OVERRIDE.get(nk, names.index[0])
    idx = g.index
    df.loc[idx, "institution_key"] = sorted(keys)[0]
    df.loc[idx, "institute_name"] = best
    merged += len(keys) - 1
note("merge_keys", merged, "institutions that were split across two keys re-joined")

# 4. City clean-ups --------------------------------------------------------------------------
CITY = {"Rajasthan": "Jaipur", "Select City": "Shimla", "Anantpur": "Anantapur", "Perintalmanna": "Perinthalmanna"}
c0 = df.city.copy()
df["city"] = df.city.replace(CITY)
df["city"] = df.city.map(lambda c: c.split(",")[-1].strip() if isinstance(c, str) and "," in c else c)
note("city", (df.city != c0).sum(), "placeholder/state-as-city values and 'locality, district' strings fixed")

# 5. weight_scheme: fill only where an exact fit is proven from the data --------------------
FILL = {("Architecture", 2018): "30/30/20/10/10", ("Architecture", 2019): "40/20/20/10/10",
        ("Law", 2018): "40/15/25/10/10", ("Law", 2019): "40/15/25/10/10"}
df["weight_scheme_basis"] = np.where(df.weight_scheme.notna(), "estimated_from_data", None)
filled = 0
for (cat, yr), ws in FILL.items():
    m = (df.category == cat) & (df.year == yr) & df.weight_scheme.isna() & df.TLR.notna()
    w = np.array([float(v) / 100 for v in ws.split("/")])
    calc = df.loc[m, ["TLR", "RPC", "GO", "OI", "PR"]].values @ w
    assert np.abs(calc - df.loc[m, "score"].values).max() < 0.05, (cat, yr)   # proof of fit
    df.loc[m, "weight_scheme"] = ws; df.loc[m, "weight_scheme_basis"] = "verified_fit_small_sample"; filled += m.sum()
note("weight_scheme", filled, "Architecture/Law 2018-19 filled; recomputed score matches within 0.05")

# 6. Recompute every derived column ----------------------------------------------------------
ex = df.rank_status.eq("exact")
df["is_top10"] = ex & (df["rank"] <= 10)
df["is_top50"] = ex & (df["rank"] <= 50)
df["is_top100"] = (ex & (df["rank"] <= 100)) | (df.rank_status.eq("band") & (df.rank_high <= 100))
df["first_year_in_category"] = df.groupby(["institution_key", "category"]).year.transform("min")
df["category_first_year"] = df.groupby("category").year.transform("min")
df["is_new_entry"] = (df.year == df.first_year_in_category) & (df.year != df.category_first_year)
df["categories_in_year"] = df.groupby("year").category.transform("nunique")

# 7. Helper columns for analysis (no imputation) ---------------------------------------------
df["has_score"] = df.score.notna()
df["has_components"] = df[["TLR", "RPC", "GO", "OI", "PR"]].notna().all(axis=1)
df["score_comparison_group"] = np.where(df.weight_scheme.notna(), df.category + " | " + df.weight_scheme, None)

# 8. Validation -------------------------------------------------------------------------------
assert not df.duplicated(["institution_key", "year", "category"]).any()
assert df.groupby("institution_key").institute_name.nunique().max() == 1
assert df.groupby("institute_name").institution_key.nunique().max() == 1
assert df.rank_status.isin(["exact", "band", "not_published", "withdrawn"]).all()
assert df.loc[df.rank_status != "exact", "rank"].isna().all() and df.loc[ex, "rank"].notna().all()
assert df.loc[df.rank_status.isin(["not_published", "withdrawn"]), "rank_value"].isna().all()
assert df.loc[df.is_rank_band, "score"].isna().all()
assert not ((df.score < 0) | (df.score > 100)).any()
df.to_csv(OUT, index=False)
pd.DataFrame(log, columns=["step", "rows_changed", "detail"]).to_csv(LOG, index=False)
print("Wrote", OUT, df.shape, "| institutions:", df.institution_key.nunique())