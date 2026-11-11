#!/usr/bin/env python3
"""
STEP 2 - PREPROCESSING: clean + standardise + merge raw_extracted.csv into the master dataset.

    python 02_preprocess.py

Needs: pip install pandas pyarrow     (run 01_extract.py first)
Optional: manual_aliases.csv (columns raw_name,canonical_name) to force institution-name fixes.

Outputs: output/nirf_master.csv, output/data_dictionary.csv/.md, output/score_weights_estimated.csv
(deliverables) and output/audit/ (aliases, duplicate review, quality report, cleaning summary).
"""
import re
import sys
import difflib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
RAW_IN = BASE / "raw_extracted.csv"
OUT = BASE / "output"

# numeric helper (also used in step 1)
def _num(x):
    if x is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else None


# --------------------------------------------------------------------------------------
# 3. CLEANING
# --------------------------------------------------------------------------------------
STATE_MAP = {
    "orissa": "Odisha", "pondicherry": "Puducherry", "uttaranchal": "Uttarakhand",
    "nct of delhi": "Delhi", "new delhi": "Delhi", "delhi ncr": "Delhi", "delhi": "Delhi",
    "jammu and kashmir": "Jammu and Kashmir", "jammu & kashmir": "Jammu and Kashmir",
    "j&k": "Jammu and Kashmir", "jammu kashmir": "Jammu and Kashmir",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra & nagar haveli and daman & diu": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "chattisgarh": "Chhattisgarh", "tamilnadu": "Tamil Nadu", "tamil nadu": "Tamil Nadu",
    "telengana": "Telangana", "telangana": "Telangana", "uttarpradesh": "Uttar Pradesh",
    "west bengal": "West Bengal", "westbengal": "West Bengal", "madhyapradesh": "Madhya Pradesh",
    "himachalpradesh": "Himachal Pradesh", "arunachalpradesh": "Arunachal Pradesh",
    "andaman and nicobar": "Andaman and Nicobar Islands", "andaman nicobar": "Andaman and Nicobar Islands",
    "andaman and nicobar island": "Andaman and Nicobar Islands", "a and n islands": "Andaman and Nicobar Islands",
}
VALID_STATES = {
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
}

# raw label (lower, no spaces/punct) -> (canonical category, group)
CATEGORY_MAP = {
    "overall": "Overall", "university": "University", "universities": "University",
    "college": "College", "colleges": "College", "research": "Research",
    "researchinstitutions": "Research", "engineering": "Engineering",
    "management": "Management", "pharmacy": "Pharmacy", "medical": "Medical",
    "dental": "Dental", "law": "Law", "architecture": "Architecture",
    "architectureandplanning": "Architecture", "architectureplanning": "Architecture",
    "agriculture": "Agriculture", "agricultureandalliedsectors": "Agriculture",
    "innovation": "Innovation", "openuniversity": "Open University",
    "openuniversities": "Open University", "statepublicuniversity": "State Public University",
    "statepublicuniversities": "State Public University", "skilluniversity": "Skill University",
    "skilluniversities": "Skill University",
}
CATEGORY_GROUP = {  # helps the dashboard offer sensible filters
    "Overall": "General", "University": "General", "College": "General", "Research": "General",
    "Open University": "General", "State Public University": "General", "Skill University": "General",
    "Innovation": "General",
}  # every other category is treated as a "Discipline"

ABBREV = [  # applied to the *matching key* only, never to the displayed name
    (r"\bindian institute of technology\b", "iit"),
    (r"\bindian institute of management\b", "iim"),
    (r"\bnational institute of technology\b", "nit"),
    (r"\bindian institute of science education and research\b", "iiser"),
    (r"\bindian institute of information technology\b", "iiit"),
    (r"\bindian institute of science\b", "iisc"),
    (r"\ball india institute of medical sciences\b", "aiims"),
    (r"\bunivers(?:ity|al)\b", "univ"), (r"\binstitute\b", "inst"), (r"\bisntitute\b", "inst"),
    (r"\btechnology\b", "tech"), (r"\btechnolgy\b", "tech"), (r"\bengineering\b", "engg"),
    (r"\bdeemed to be\b", "deemed"), (r"\bdeemed\b", "deemed"),
    (r"\bst\.?\b", "saint"),
    (r"\bgovernment\b", "govt"), (r"\bgoverment\b", "govt"),          # spelling variants
    (r"\bbangalore\b", "bengaluru"), (r"\btrichur\b", "thrissur"), (r"\btiruchirapalli\b", "tiruchirappalli"),
    (r"\bmalviya\b", "malaviya"), (r"\bmody\b", "modi"),
]


def clean_text(s):
    if pd.isna(s):
        return None
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip(" \t\r\n,;")
    s = re.sub(r"\s*&\s*", " & ", s)
    return s or None


def name_key(name, city=None):
    """
    Aggressive normalisation used only for MATCHING variants of one institution (never displayed).
    Handles apostrophes/backticks, dots, initials (G. M. R. = GMR), PIN codes, 'Dist.', Govt/Government,
    Bangalore/Bengaluru, a trailing city name (if `city` is given) and a trailing 'University'.
    """
    if not name:
        return ""
    k = unicodedata.normalize("NFKC", str(name)).lower()
    k = re.sub(r"\(.*?\)", " ", k)                 # drop bracketed qualifiers e.g. (Deemed to be University)
    k = re.sub(r"[`'\u2019]s\b", "s", k)            # Joseph`s / Joseph's -> josephs
    k = re.sub(r"[`'\u2019]", " ", k)
    k = k.replace("&", " and ").replace(".", "")
    k = re.sub(r"[-\u2013,/]", " ", k)
    k = re.sub(r"\b\d{3}\s?\d{3}\b", " ", k)       # PIN codes
    k = re.sub(r"\bdist(?:rict)?\b", " ", k)
    k = re.sub(r"[^a-z0-9 ]", " ", k)
    k = re.sub(r"\s+", " ", k).strip()
    for pat, rep in ABBREV:
        k = re.sub(pat, rep, k)
    k = re.sub(r"\band\b|\bthe\b|\bof\b", " ", k)
    k = re.sub(r"\b([a-z])\s+(?=[a-z]\b)", r"\1", k)   # "g m r" -> "gmr"
    k = re.sub(r"\s+", " ", k).strip()
    if city and isinstance(city, str):
        c = re.sub(r"[^a-z ]", "", city.lower()).strip()
        for pat, rep in ABBREV:
            c = re.sub(pat, rep, c)
        if c and k.endswith(" " + c) and len(k) > len(c) + 8:
            k = k[: -len(c) - 1]                      # "... College, Kopargaon" == "... College"
    if k.endswith(" univ") and len(k.split()) > 2:
        k = k[:-5]                                    # "Symbiosis International University" == "Symbiosis International"
    return re.sub(r"\s+", " ", k).strip()


def std_state(s):
    s = clean_text(s)
    if not s:
        return None
    low = re.sub(r"\s+", " ", s.lower().replace("&", "and")).strip()
    if low in STATE_MAP:
        return STATE_MAP[low]
    t = s.title().replace(" And ", " and ")
    if t in VALID_STATES:
        return t
    match = difflib.get_close_matches(t, VALID_STATES, n=1, cutoff=0.85)
    return match[0] if match else t


def std_category(c):
    c = clean_text(c) or "Unknown"
    k = re.sub(r"[^a-z]", "", c.lower().replace("ranking", ""))
    return CATEGORY_MAP.get(k, c.title())


def parse_rank(r):
    """Return (rank_low, rank_high, is_band). Handles '12', '101-150', '151 - 200', '101-150 (Band)'."""
    if pd.isna(r):
        return (None, None, False)
    s = str(r).strip()
    m = re.match(r"^\D*(\d+)\s*[-–to]+\s*(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), True)
    m = re.search(r"\d+", s)
    if m:
        v = int(m.group())
        return (v, v, False)
    return (None, None, False)



STATE_CODE = {  # ISO 3166-2:IN codes, handy for joining to state-level GeoJSON/choropleth maps
    "Andhra Pradesh": "AP", "Arunachal Pradesh": "AR", "Assam": "AS", "Bihar": "BR", "Chhattisgarh": "CT",
    "Goa": "GA", "Gujarat": "GJ", "Haryana": "HR", "Himachal Pradesh": "HP", "Jharkhand": "JH",
    "Karnataka": "KA", "Kerala": "KL", "Madhya Pradesh": "MP", "Maharashtra": "MH", "Manipur": "MN",
    "Meghalaya": "ML", "Mizoram": "MZ", "Nagaland": "NL", "Odisha": "OD", "Punjab": "PB",
    "Rajasthan": "RJ", "Sikkim": "SK", "Tamil Nadu": "TN", "Telangana": "TG", "Tripura": "TR",
    "Uttar Pradesh": "UP", "Uttarakhand": "UT", "West Bengal": "WB",
    "Andaman and Nicobar Islands": "AN", "Chandigarh": "CH",
    "Dadra and Nagar Haveli and Daman and Diu": "DH", "Delhi": "DL", "Jammu and Kashmir": "JK",
    "Ladakh": "LA", "Lakshadweep": "LD", "Puducherry": "PY",
}

NAME_TYPOS = {"Universtiy": "University", "Enterprenurship": "Entrepreneurship", "Isntitute": "Institute",
              "Technolgy": "Technology", "Goverment": "Government"}
CITY_FIX = {"gurgaon": "Gurugram", "calcutta": "Kolkata", "bombay": "Mumbai", "madras": "Chennai",
            "bangalore": "Bengaluru", "trichur": "Thrissur", "poona": "Pune", "cochin": "Kochi",
            "tiruchirapalli": "Tiruchirappalli", "visakapatnam": "Visakhapatnam", "banaras": "Varanasi"}
CATEGORY_PRIORITY = {"Overall": 0, "University": 1, "College": 2}   # preferred source of an institution's display name


def display_name(n):
    """Cosmetic clean-up of the name shown in the dashboard (matching uses name_key, not this)."""
    if not isinstance(n, str):
        return n
    n = re.sub(r"[`’]", "'", n)
    for bad, good in NAME_TYPOS.items():
        n = n.replace(bad, good)
    n = re.sub(r"[\s,\-–]*\b\d{3}\s?\d{3}\b\.?\s*$", "", n)      # trailing PIN code
    n = re.sub(r"\s+,", ",", n)
    return re.sub(r"\s+", " ", n).strip(" ,-")


def std_city(c):
    c = clean_text(c)
    if not c:
        return None
    c = re.sub(r"(?i)^(district|dist\.?)\s+", "", c)
    c = re.sub(r"(?i)^sector[-\s]*\d+\s*,?\s*", "", c)
    c = re.sub(r"(?i)\s+district$", "", c)
    c = re.sub(r"(?i),\s*dist\.?\s+\w+$", "", c)                          # "Hubli, Dist. Dharwad" -> "Hubli"
    c = re.sub(r"(?:[-\s]*\d+)+$", "", c)                             # "Calcutta-16", "Kumbakonam - 612 001"
    c = re.sub(r"(?i)\s+dist\.?$", "", c)
    parts = [p.strip() for p in c.split(",") if p.strip()]
    if len(parts) > 1 and (parts[-1].title() in VALID_STATES or parts[-1].lower() in STATE_MAP):
        parts = parts[:-1]                                            # "Gurugram, Haryana" -> "Gurugram"
    parts = [CITY_FIX.get(p.lower(), p) for p in parts]
    return ", ".join(parts).title() if parts else None


def _typo_match(a, b):
    """True only if two keys differ by ONE long word that looks like a typo (not IIT vs IIM, RMD vs RMK)."""
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb) or len(ta) < 3:
        return False
    diff = [(x, y) for x, y in zip(ta, tb) if x != y]
    if len(diff) != 1:
        return False
    x, y = diff[0]
    return len(x) >= 6 and len(y) >= 6 and difflib.SequenceMatcher(None, x, y).ratio() >= 0.8


def build_alias_table(df):
    """
    Map every row key -> ONE institution.
    Keys are "<state>|<normalised name>", so same-named institutions in different states never merge.
    Merge rules: (1) identical key; (2) identical NIRF institute_id (NIRF's own identity, may cross states,
    e.g. Amrita School of Pharmacy = Amrita Vishwa Vidyapeetham); (3) single-word typos within a state.
    No merge is allowed if both sides appear in the same year+category.
    The display name comes from the broadest category (Overall > University > College > others).
    """
    df = df.copy()
    df["key"] = df["_key"]
    keys = list(df["key"].unique())
    parent = {k: k for k in keys}
    occ = {k: set(zip(g["year"], g["category"])) for k, g in df.groupby("key")}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            occ[ra] = occ[ra] | occ[rb]

    if "institute_id" in df.columns:                                   # (2) NIRF id
        for _, g in df.dropna(subset=["institute_id"]).groupby("institute_id"):
            ks = list(g["key"].unique())
            for k in ks[1:]:
                union(ks[0], k)
    for _, g in df.groupby("_kfull"):                                  # same full name, only the city differs
        ks = list(g["key"].unique())                                   # (Ayodhya/Faizabad, Mohali/SAS Nagar, typos)
        for k in ks[1:]:
            ra, rb = find(ks[0]), find(k)
            if ra != rb and not (occ[ra] & occ[rb]):
                union(ks[0], k)
    by_state = {}
    for k in keys:
        by_state.setdefault(k.split("|", 1)[0], []).append(k)
    for st, ks in by_state.items():                                    # (3) typos within a state
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                ra, rb = find(a), find(b)
                if ra == rb or occ[ra] & occ[rb]:
                    continue
                if _typo_match(a.split("|", 1)[1], b.split("|", 1)[1]):
                    union(a, b)

    df["_root"] = df["key"].map(find)
    df["_prio"] = df["category"].map(lambda c: CATEGORY_PRIORITY.get(c, 3))
    best = (df.groupby(["_root", "institute_name"]).agg(p=("_prio", "min"), n=("year", "size"), last=("year", "max"))
              .reset_index().sort_values(["_root", "p", "n", "last"], ascending=[True, True, False, False])
              .drop_duplicates("_root").set_index("_root")["institute_name"].map(display_name))
    root_state = df.groupby("_root")["state"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "")
    names = best.to_dict()
    dup = pd.Series(names).value_counts()
    for r in [r for r, n in names.items() if dup[names[r]] > 1]:       # same display name, different institutions
        names[r] = f"{names[r]} ({root_state[r]})" if root_state[r] else names[r]
    root_city = df.groupby("_root")["city"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "")
    dup = pd.Series(names).value_counts()
    for r in [r for r, n in names.items() if dup[names[r]] > 1]:       # still identical (same state) -> add the city
        names[r] = f"{names[r]} [{root_city[r]}]" if root_city[r] else names[r]
    ids, i = {}, 0
    for r in sorted(names, key=lambda r: (names[r], r)):
        i += 1
        ids[r] = f"INST{i:04d}"
    canon = {k: names[find(k)] for k in keys}
    key_id = {k: ids[find(k)] for k in keys}
    aliases = df.groupby(["institute_name", "key"]).size().reset_index(name="n_rows")
    aliases["canonical_name"] = aliases["key"].map(canon)
    aliases["institution_key"] = aliases["key"].map(key_id)
    aliases["was_changed"] = aliases["institute_name"] != aliases["canonical_name"]
    return aliases.sort_values(["canonical_name", "institute_name"]), canon, key_id


def derive_band_ranges(df):
    """
    Band pages list institutions without an exact rank. Each band follows on from the last exact rank of
    that year+category: page '...150' first, then '...200'. Width = number of institutions on the page,
    rounded to the nearest 25 (NIRF bands are 25/50 wide). Examples: Overall 100 exact -> 101-150, 151-200;
    Engineering 2020 (200 exact) -> 201-250, 251-300; Management 2019 (75 exact) -> 76-100.
    These ranges are DERIVED (rank_band_source = 'derived'), verify a few on nirfindia.org.
    """
    exact_max = df[df["rank"].notna()].groupby(["year", "category"])["rank"].max()
    band = df["is_rank_band"] & df["rank_low"].isna()
    for (y, c), g in df[band].groupby(["year", "category"]):
        if (y, c) not in exact_max.index:
            continue
        start = int(exact_max[(y, c)])
        up = int(np.ceil(start / 25.0) * 25)
        if 0 < up - start <= 3:      # ties can leave the highest exact rank a little short (e.g. 99 -> bands start at 101)
            start = up
        pages = sorted(g["source_page"].unique(),
                       key=lambda p: int(re.search(r"(\d+)$", p).group(1)) if re.search(r"(\d+)$", str(p)) else 0)
        for page in pages:
            idx = g.index[g["source_page"] == page]
            width = max(25, int(round(len(idx) / 25.0)) * 25)
            df.loc[idx, "rank_low"] = start + 1
            df.loc[idx, "rank_high"] = start + width
            start += width
    return df


def estimate_weights(df, params=("TLR", "RPC", "GO", "OI", "PR")):
    """Score = weighted sum of the parameter scores. Weights changed by category/year -> estimate them by
    least squares per category-year (R^2 = 1.000 means the weights are exact)."""
    r = df[df[list(params) + ["score"]].notna().all(axis=1)]
    rows = []
    for (c, y), g in r.groupby(["category", "year"]):
        if len(g) < 20:
            continue
        X, yv = g[list(params)].values, g["score"].values
        w = np.linalg.lstsq(X, yv, rcond=None)[0]
        ss = ((yv - yv.mean()) ** 2).sum()
        r2 = 1 - ((yv - X @ w) ** 2).sum() / ss if ss else np.nan
        rows.append([c, int(y), len(g)] + [round(float(x), 2) for x in w] + [round(float(r2), 4)])
    W = pd.DataFrame(rows, columns=["category", "year", "n_rows"] + list(params) + ["r2"])
    W["weight_scheme"] = W[list(params)].apply(lambda t: "/".join(f"{int(round(v * 100))}" for v in t), axis=1)
    return W


def clean(raw):
    df = raw.copy()
    report = {"rows_raw": len(df)}
    for c in ["institute_id", "institute_name", "city", "state", "category_original"]:
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].map(clean_text)
    df["year"] = pd.to_numeric(df["year"].astype(str).str.extract(r"(20\d{2})")[0], errors="coerce").astype("Int64")
    df = df[df["year"].between(2017, 2025)]
    df["category_original"] = df["category_original"].fillna("Unknown")
    df["category"] = df["category_original"].map(std_category)
    df["category_group"] = df["category"].map(lambda c: CATEGORY_GROUP.get(c, "Discipline"))
    df["state_original"] = df["state"]
    df["state"] = df["state"].map(std_state)
    df["state_code"] = df["state"].map(STATE_CODE)
    df["city_original"] = df["city"]
    df["city"] = df["city"].map(std_city)

    for c in ["score", "TLR", "RPC", "GO", "OI", "PR", "SS"]:
        if c in df.columns:
            df[c] = df[c].map(_num)
    if "rank" not in df.columns:
        df["rank"] = None
    df["rank_original"] = df["rank"].astype("string")                 # rank text exactly as published
    rk = df["rank"].map(parse_rank)
    df["rank_low"] = [r[0] for r in rk]
    df["rank_high"] = [r[1] for r in rk]
    df["is_rank_band"] = [r[2] for r in rk]
    if "is_band_page" in df.columns:    # rows from the "rank band" pages: listed but no exact rank/score
        band_only = df["is_band_page"].astype(str).str.lower().eq("true") & df["rank_low"].isna()
        df["is_rank_band"] = df["is_rank_band"] | band_only
    df["rank"] = df["rank_low"].where(~df["is_rank_band"])           # exact rank only

    for c in ["score", "TLR", "RPC", "GO", "OI", "PR"]:                # values must lie in [0, 100]
        if c in df.columns:
            bad = ~df[c].between(0, 100) & df[c].notna()
            report[f"{c}_out_of_range_set_null"] = int(bad.sum())
            df.loc[bad, c] = None

    df = df.dropna(subset=["institute_name", "year"])
    manual = BASE / "manual_aliases.csv"   # optional: raw_name,canonical_name (your corrections win)
    if manual.exists():
        m = pd.read_csv(manual).dropna()
        df["institute_name"] = df["institute_name"].replace(
            dict(zip(m.raw_name.map(clean_text), m.canonical_name.map(clean_text))))
    df["institute_name_original"] = df["institute_name"]
    # Same name, different institutions in ONE year+category (e.g. two "College of Engineering"): add the city.
    combo = df.drop_duplicates(["year", "category", "institute_name", "institute_id", "city"])
    cnt = combo.groupby(["year", "category", "institute_name"]).size()
    ambiguous = set(cnt[cnt > 1].reset_index()["institute_name"])
    m_amb = df["institute_name"].isin(ambiguous) & df["city"].notna()
    df.loc[m_amb, "institute_name"] = df.loc[m_amb, "institute_name"] + " - " + df.loc[m_amb, "city"]
    report["same_name_institutions_split_by_city"] = len(ambiguous)

    # Row key = "<state>|<normalised name>". If two different names collapse to one key inside the same
    # year+category they are different institutions -> fall back to the key WITH the city words.
    st = df["state"].fillna("?")
    df["_key"] = [f"{s}|{name_key(n, c)}" for s, n, c in zip(st, df["institute_name"], df["city"])]
    chk = df.drop_duplicates(["_key", "year", "category", "institute_name", "city"])
    clash_keys = set(chk.loc[chk.duplicated(["_key", "year", "category"], keep=False), "_key"])
    cm = df["_key"].isin(clash_keys)
    df.loc[cm, "_key"] = [f"{s}|{name_key(n)}" for s, n in zip(st[cm], df.loc[cm, "institute_name"])]
    report["keys_kept_separate_due_to_same_year_clash"] = len(clash_keys)
    df["_kfull"] = [f"{s}|{name_key(n)}" for s, n in zip(st, df["institute_name"])]
    aliases, canon, key_id = build_alias_table(df)
    df["institute_name"] = df["_key"].map(canon)
    df["institution_key"] = df["_key"].map(key_id)

    for c in ["state", "city"]:                # fill missing from other rows of the same institution
        fill = df.dropna(subset=[c]).groupby("institution_key")[c].agg(lambda s: s.mode().iat[0])
        miss = df[c].isna()
        report[f"{c}_filled_from_other_years"] = int(miss.sum())
        df.loc[miss, c] = df.loc[miss, "institution_key"].map(fill)
        report[f"{c}_filled_from_other_years"] -= int(df[c].isna().sum())
    df["state_code"] = df["state"].map(STATE_CODE)

    # duplicates: same institution, year, category. Keep the most complete row; prefer exact rank over band.
    param_cols = [c for c in ["score", "TLR", "RPC", "GO", "OI", "PR", "rank"] if c in df.columns]
    df["_complete"] = df[param_cols].notna().sum(axis=1) + (~df["is_rank_band"]).astype(int)
    before = len(df)
    df = (df.sort_values("_complete", ascending=False)
            .drop_duplicates(["institution_key", "year", "category"], keep="first")
            .drop(columns=["_complete", "_key", "_kfull"]))
    report["duplicates_removed"] = before - len(df)
    df = df.reset_index(drop=True)

    # ---- rank information -------------------------------------------------------------------
    df["rank_status"] = np.where(df["rank"].notna(), "exact", np.where(df["is_rank_band"], "band", "not_published"))
    wd = df["rank_original"].fillna("").str.contains("withdraw", case=False)
    df.loc[(df["rank_status"] == "not_published") & wd, "rank_status"] = "withdrawn"
    # an unranked row WITH a score whose position by score is a free slot in the published ranks
    cand = df[(df["rank_status"] == "not_published") & df["score"].notna()]
    for idx, row in cand.iterrows():
        g = df[(df["year"] == row["year"]) & (df["category"] == row["category"]) & df["rank"].notna()]
        pos = int((g["score"] > row["score"]).sum()) + 1
        if pos not in set(g["rank"].astype(int)):
            df.loc[idx, ["rank", "rank_low", "rank_high"]] = pos
            df.loc[idx, "rank_status"] = "inferred_from_score"
    report["ranks_inferred_from_score"] = int((df["rank_status"] == "inferred_from_score").sum())
    df = derive_band_ranges(df)
    df["rank_band_source"] = np.where(df["rank_status"] == "band", "derived", None)
    isb = df["rank_status"] == "band"
    df["rank_band_label"] = None
    df.loc[isb & df["rank_low"].notna(), "rank_band_label"] = (
        df.loc[isb, "rank_low"].astype("Int64").astype(str) + "-" + df.loc[isb, "rank_high"].astype("Int64").astype(str))
    df["rank_midpoint"] = (df["rank_low"] + df["rank_high"]) / 2
    df["rank_value"] = df["rank"].fillna(df["rank_midpoint"])
    report["band_rows_without_range"] = int((isb & df["rank_low"].isna()).sum())

    # ---- derived flags ----------------------------------------------------------------------
    df["is_top10"] = df["rank"].le(10)
    df["is_top50"] = df["rank"].le(50)
    df["is_top100"] = df["rank"].le(100) | (df["rank_high"].le(100) & df["is_rank_band"])
    first = df.groupby(["category", "institution_key"])["year"].transform("min")
    df["first_year_in_category"] = first
    df["is_new_entry"] = (df["year"] == first) & (df["year"] > df.groupby("category")["year"].transform("min"))
    df["category_first_year"] = df.groupby("category")["year"].transform("min")
    df["categories_in_year"] = df["year"].map(df.groupby("year")["category"].nunique())
    W = estimate_weights(df)
    df = df.merge(W[["category", "year", "weight_scheme"]], on=["category", "year"], how="left")
    df.attrs["weights"] = W

    order = ["institution_key", "institute_name", "institute_id", "institute_name_original", "year",
             "category", "category_original", "category_group", "city", "city_original", "state", "state_code",
             "state_original", "rank", "rank_value", "rank_status", "rank_original", "rank_low", "rank_high",
             "is_rank_band", "rank_band_label", "rank_band_source", "rank_midpoint",
             "score", "TLR", "RPC", "GO", "OI", "PR", "SS", "weight_scheme",
             "is_top10", "is_top50", "is_top100", "is_new_entry", "first_year_in_category",
             "category_first_year", "categories_in_year", "source_page"]
    order = [c for c in order if c in df.columns]
    df = df[order].sort_values(["year", "category", "rank_low", "institute_name"]).reset_index(drop=True)
    report["rows_clean"] = len(df)
    return df, aliases, report


# ---------------------------------------------------------------- 4. REPORTS + DATA DICTIONARY

DICT = {
    "institution_key": ("Stable internal ID for one institution across all years and categories (our own, not NIRF's). Built from state + normalised name + NIRF institute_id.", "string", "Join key for trends/mobility"),
    "institute_name": ("Standardised (canonical) institution name, taken from the broadest category (Overall > University > College > others). Same-name institutions in different states carry the state in brackets.", "string", "Use for all charts and filters"),
    "institute_id": ("Institute ID printed by NIRF (2019+ format IR-E-U-0456; older years use other formats).", "string", "Optional; many missing for band rows"),
    "institute_name_original": ("Name exactly as published that year, after whitespace/unicode cleanup.", "string", "Audit only"),
    "year": ("NIRF ranking year.", "int", "2017-2025"),
    "category": ("Standardised ranking category (Overall, Engineering, Pharmacy, ...).", "string", "Filter / facet. Never add up or average across categories: the same institution is listed in several categories in one year"),
    "category_original": ("Category label as published on the source page.", "string", "Keeps historical definitions"),
    "category_group": ("General (Overall, University, College, Research, ...) vs Discipline (Engineering, Pharmacy, ...).", "string", "Helps compare like with like"),
    "city": ("Cleaned city (no 'District', PIN/sector numbers or trailing state; Gurgaon->Gurugram, Calcutta->Kolkata, ...).", "string", "Prefer state for maps"),
    "city_original": ("City as published.", "string", "Audit only"),
    "state": ("Standardised state/UT name (e.g. Orissa -> Odisha). It is the location of that year's listing.", "string", "Use for geospatial analysis (state-level maps)"),
    "state_code": ("ISO 3166-2:IN two-letter state/UT code (TN, KA, DL, ...).", "string", "Join key for state GeoJSON / choropleth"),
    "state_original": ("State as published.", "string", "Audit only"),
    "rank": ("Exact NIRF rank (1 = best). Null when rank_status is band, not_published or withdrawn. Ties share a rank.", "float", "Never imputed except rank_status = inferred_from_score"),
    "rank_value": ("rank if exact, otherwise rank_midpoint of the band. Use when every institution needs a number (mobility, improvement, volatility).", "float", "Banded rows are approximate (+/- half a band)"),
    "rank_status": ("exact = published rank; band = listed only in a rank band; not_published = listed with no rank (Innovation 2023); withdrawn = 'Rank Withdrawn'; inferred_from_score = rank missing in the source but the score fits exactly one free slot in the published sequence.", "string", "Filter on this instead of testing rank for null"),
    "rank_original": ("Rank text exactly as published (e.g. '21A', 'Rank Withdrawn').", "string", "Audit only"),
    "rank_low": ("Lower bound of the rank (equals rank for exact ranks; start of the band for banded rows).", "float", ""),
    "rank_high": ("Upper bound of the rank (equals rank for exact ranks; end of the band for banded rows).", "float", ""),
    "is_rank_band": ("True if the institution is listed only in a rank band (no exact rank, score or parameters).", "bool", ""),
    "rank_band_label": ("Band as 'low-high', e.g. '101-150' or '201-250'.", "string", "DERIVED, see rank_band_source"),
    "rank_band_source": ("'derived': the band starts after the last exact rank of that year+category; pages '...150' then '...200'; width = institutions on the page rounded to 25.", "string", "Verify a few bands on nirfindia.org"),
    "rank_midpoint": ("(rank_low + rank_high)/2. Filled for exact ranks and for bands.", "float", ""),
    "score": ("NIRF overall score (0-100) for that category and year. Null for band rows and a few small categories.", "float", ""),
    "TLR": ("Teaching, Learning & Resources score (0-100).", "float", "Parameter 1"),
    "RPC": ("Research and Professional Practice score (0-100). Called 'RP' in older years.", "float", "Parameter 2"),
    "GO": ("Graduation Outcomes score (0-100).", "float", "Parameter 3"),
    "OI": ("Outreach and Inclusivity score (0-100).", "float", "Parameter 4"),
    "PR": ("Perception score (0-100).", "float", "Parameter 5"),
    "SS": ("Student Strength score, only if the source year published it.", "float", "Mostly null"),
    "weight_scheme": ("Weights (%) of TLR/RPC/GO/OI/PR in the score for that category and year, estimated from the data (R2 = 1). E.g. 30/30/20/10/10.", "string", "Scores are only strictly comparable under the same scheme (see score_weights_estimated.csv)"),
    "is_top10": ("rank <= 10 within its category and year.", "bool", "Derived"),
    "is_top50": ("rank <= 50 within its category and year.", "bool", "Derived"),
    "is_top100": ("rank <= 100 (banded rows count only if the whole band lies within 100).", "bool", "Derived"),
    "is_new_entry": ("First year the institution appears in that category (excluding the category's first year).", "bool", "Derived; for 'newly appearing' questions"),
    "first_year_in_category": ("First year the institution appears in that category.", "int", "Derived"),
    "category_first_year": ("First year the category exists in the data (e.g. Dental, Innovation start later than 2017).", "int", "Derived; handles category changes"),
    "categories_in_year": ("Number of categories published that year (6 in 2017, more later).", "int", "Derived"),
    "source_page": ("Source file / page name the row was taken from.", "string", "Lineage"),
}

ANALYSIS_NOTES = [
    "Do not add up or average across categories: an institution has about 1.5 rows per year (Overall, University, Engineering, ...). Always filter or group by category.",
    "Band rows (about 37%) have no exact rank, score or parameters. Use rank_value for rank-movement questions and say that banded ranks are approximate.",
    "Band ranges are derived from the last exact rank of each category-year, not read from the site: check a few against nirfindia.org.",
    "The score weights changed by category and year (e.g. College 2017 vs 2018+). Compare scores across years only within one weight_scheme, or note the difference.",
    "An institution can appear in two states only when NIRF's own institute_id links the listings (e.g. Amrita School of Pharmacy, Kochi = Amrita Vishwa Vidyapeetham). Use the row's state for state-level maps.",
    "Research, Innovation, Open University and Skill University use different parameters/scoring: TLR..PR are empty for them.",
]


def review_possible_duplicates(df, cutoff=0.8):
    """Pairs of *different* canonical names in the same state that look similar -> check by hand,
    then add the fix to manual_aliases.csv (raw_name,canonical_name) and re-run `build`."""
    inst = df.groupby("institute_name").agg(state=("state", lambda s: s.mode().iat[0] if s.notna().any() else None),
                                            years=("year", lambda s: ",".join(map(str, sorted(set(s)))))).reset_index()
    out = []
    for st, g in inst.groupby("state", dropna=False):
        rows = g.to_dict("records")
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                ka, kb = name_key(a["institute_name"]), name_key(b["institute_name"])
                r = difflib.SequenceMatcher(None, ka, kb).ratio()
                if r >= cutoff or ka.startswith(kb) or kb.startswith(ka):
                    out.append({"state": st, "name_a": a["institute_name"], "years_a": a["years"],
                                "name_b": b["institute_name"], "years_b": b["years"], "similarity": round(r, 3)})
    return pd.DataFrame(out, columns=["state", "name_a", "years_a", "name_b", "years_b", "similarity"])


def write_reports(df, aliases, report):
    OUT.mkdir(exist_ok=True)
    AUDIT = OUT / "audit"
    AUDIT.mkdir(exist_ok=True)
    df.to_csv(OUT / "nirf_master.csv", index=False)
    try:
        df.to_parquet(AUDIT / "nirf_master.parquet", index=False)
    except Exception:
        pass
    aliases.to_csv(AUDIT / "institution_aliases.csv", index=False)
    review_possible_duplicates(df).to_csv(AUDIT / "possible_duplicates_review.csv", index=False)
    W = df.attrs.get("weights", estimate_weights(df))
    W.to_csv(OUT / "score_weights_estimated.csv", index=False)

    q = (df.groupby(["year", "category"]).agg(
        institutions=("institution_key", "nunique"), rows=("institution_key", "size"),
        exact_rank_rows=("rank", lambda s: int(s.notna().sum())),
        band_rows=("is_rank_band", "sum"),
        missing_score=("score", lambda s: int(s.isna().sum())),
        missing_state=("state", lambda s: int(s.isna().sum()))).reset_index())
    for p in ["TLR", "RPC", "GO", "OI", "PR"]:
        if p in df.columns:
            q[f"missing_{p}"] = df.groupby(["year", "category"])[p].apply(lambda s: int(s.isna().sum())).values
    q.to_csv(AUDIT / "data_quality_report.csv", index=False)
    pd.Series(report).to_csv(AUDIT / "cleaning_summary.csv", header=["value"])

    rows = []
    for col in df.columns:
        desc, typ, note = DICT.get(col, ("", str(df[col].dtype), ""))
        ex = df[col].dropna().iloc[0] if df[col].notna().any() else ""
        rows.append({"column": col, "type": typ, "description": desc,
                     "missing_pct": round(df[col].isna().mean() * 100, 2),
                     "unique_values": df[col].nunique(), "example": ex, "notes": note})
    pd.DataFrame(rows).to_csv(OUT / "data_dictionary.csv", index=False)

    md = ["# NIRF master dataset: data dictionary\n",
          f"Rows: {len(df):,} | Years: {df.year.min()}-{df.year.max()} | Categories: {df.category.nunique()} | "
          f"Institutions: {df.institution_key.nunique():,}\n",
          "| Column | Type | Description | Missing % | Notes |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['column']} | {r['type']} | {r['description']} | {r['missing_pct']} | {r['notes']} |")
    md += ["", "## Notes for analysis", ""] + [f"- {n}" for n in ANALYSIS_NOTES]
    md += ["", "## Score weights by category and year (TLR/RPC/GO/OI/PR, %)", "",
           "Estimated from the data by least squares; R2 = 1.000 means exact.", "",
           "| Category | Weights | Years |", "|---|---|---|"]
    g = W.groupby(["category", "weight_scheme"]).agg(years=("year", lambda s: ", ".join(map(str, sorted(s)))),
                                                    r2=("r2", "min")).reset_index()
    for _, r in g.iterrows():
        md.append(f"| {r['category']} | {r['weight_scheme']} | {r['years']} |")
    (OUT / "data_dictionary.md").write_text("\n".join(md), encoding="utf-8")


def main():
    if not RAW_IN.exists():
        sys.exit("raw_extracted.csv not found - run 01_extract.py first.")
    raw = pd.read_csv(RAW_IN, dtype=str)
    df, aliases, report = clean(raw)
    write_reports(df, aliases, report)
    print(f"Master dataset: {len(df):,} rows, {df.institution_key.nunique():,} institutions, "
          f"{df.category.nunique()} categories, years {df.year.min()}-{df.year.max()}")
    print("Cleaning summary:", report)
    print(f"Outputs in {OUT}")


if __name__ == "__main__":
    main()