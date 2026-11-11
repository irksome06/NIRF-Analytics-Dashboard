#!/usr/bin/env python3
"""
STEP 1 - EXTRACTION: collect NIRF data (2017-2025) and write one raw, uncleaned table.

    python 01_extract.py scrape    # download HTML pages from nirfindia.org into raw_html/
    python 01_extract.py parse     # parse raw_html/ (+ any CSVs in raw_csv/) -> raw_extracted.csv
    python 01_extract.py all       # both

Needs: pip install pandas beautifulsoup4 lxml requests
Pages saved by scrape are named like 2024__EngineeringRanking.html. Pages you save by hand
from the browser (Ctrl+S, "Webpage, HTML only") can have any name: year and category are read
from the page heading ("India Rankings 2024: Engineering").
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
RAW_HTML = BASE / "raw_html"
RAW_CSV = BASE / "raw_csv"
RAW_OUT = BASE / "raw_extracted.csv"

YEARS = list(range(2017, 2026))

# --------------------------------------------------------------------------------------
# 1. COLLECTION
# --------------------------------------------------------------------------------------
# Page names used on nirfindia.org. Not every category exists in every year; 404s are
# skipped and recorded, so the list can safely be a superset.
CATEGORY_PAGES = [
    "OverallRanking", "UniversityRanking", "CollegeRanking", "ResearchRanking",
    "EngineeringRanking", "ManagementRanking", "PharmacyRanking", "MedicalRanking",
    "DentalRanking", "LawRanking", "ArchitectureRanking", "AgricultureRanking",
    "InnovationRanking", "OpenUniversityRanking", "StatePublicUniversityRanking",
    "SkillUniversityRanking", "ArchitectureandPlanningRanking",
    "AgricultureandAlliedSectorsRanking",
]
# URL layout changed over the years -> several templates are tried in order.
# !! Verify these in a browser once; adjust if NIRF changes its site. !!
URL_TEMPLATES = [
    "https://www.nirfindia.org/Rankings/{year}/{page}.html",
    "https://www.nirfindia.org/{year}/{page}.html",
    "https://www.nirfindia.org/nirfpdfcdn/{year}/{page}.html",
]
# Rank bands published on separate pages in some years (e.g. 101-150, 151-200).
BAND_SUFFIXES = ["", "150", "200"]


def scrape(years=YEARS, delay=1.0):
    import requests
    RAW_HTML.mkdir(exist_ok=True)
    log = []
    headers = {"User-Agent": "Mozilla/5.0 (academic project; NIRF dashboard)"}
    for year in years:
        for page in CATEGORY_PAGES:
            for suf in BAND_SUFFIXES:
                name = page.replace("Ranking", "Ranking" + suf)
                dest = RAW_HTML / f"{year}__{name}.html"
                if dest.exists():
                    continue
                got = False
                for tpl in URL_TEMPLATES:
                    url = tpl.format(year=year, page=name)
                    try:
                        r = requests.get(url, headers=headers, timeout=30)
                    except requests.RequestException as e:
                        log.append((year, name, url, f"error {e}"))
                        continue
                    if r.status_code == 200 and "<table" in r.text.lower():
                        dest.write_text(r.text, encoding="utf-8")
                        log.append((year, name, url, "ok"))
                        got = True
                        break
                    log.append((year, name, url, r.status_code))
                    time.sleep(0.2)
                if got:
                    time.sleep(delay)
    pd.DataFrame(log, columns=["year", "page", "url", "status"]).to_csv(
        BASE / "scrape_log.csv", index=False)
    print(f"Saved pages in {RAW_HTML}")


# --------------------------------------------------------------------------------------
# 2. PARSING (raw -> long table, no cleaning yet)
# --------------------------------------------------------------------------------------
PARAM_ALIASES = {
    "tlr": "TLR", "rpc": "RPC", "rp": "RPC", "go": "GO", "oi": "OI",
    "pr": "PR", "perception": "PR", "pe": "PR", "ss": "SS",  # SS = Student Strength (older years)
}


def _num(x):
    if x is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else None


def _param_key(label):
    k = re.sub(r"\(.*?\)", "", label).strip().lower()
    k = re.sub(r"[^a-z]", "", k)
    return PARAM_ALIASES.get(k)


def parse_html(html, year, page):
    """Parse one NIRF ranking page into a list of dict rows."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    category = re.sub(r"Ranking\d*$", "", page)
    band_page = bool(re.search(r"Ranking(150|200)$", page))
    rows = []
    for table in soup.find_all("table"):
        if table.find_parent("table"):
            continue  # nested detail table, handled with its parent row
        header = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
        if not any("name" in h for h in header) or not any("rank" in h or "score" in h for h in header):
            # some years use <td> headers
            first = table.find("tr")
            header = [c.get_text(" ", strip=True).lower() for c in first.find_all(["td", "th"])] if first else []
            if not any("name" in h for h in header):
                continue
        # use only the top-level header cells
        top_header = [c.get_text(" ", strip=True).lower() for c in table.find("tr").find_all(["th", "td"], recursive=False)]
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue
            tds = tr.find_all("td", recursive=False)
            if len(tds) < 3 or tr.find_all("th", recursive=False):
                continue
            cells = []
            for td in tds:
                first_text = next((s.strip() for s in td.find_all(string=True, recursive=False) if s.strip()), "")
                cells.append(first_text or td.get_text(" ", strip=True))
            rec = {"year": year, "category_original": category, "source_page": page,
                   "is_band_page": band_page}
            for i, h in enumerate(top_header[:len(cells)]):
                v = cells[i]
                if "id" in h and "name" not in h:
                    rec["institute_id"] = v
                elif "name" in h:
                    rec["institute_name"] = v
                elif "city" in h:
                    rec["city"] = v
                elif "state" in h:
                    rec["state"] = v
                elif "score" in h:
                    rec["score"] = v
                elif "rank" in h:
                    rec["rank"] = v
            # nested "score details" table: either header row + value row, or label/value pairs
            for nested in tr.find_all("table"):
                nrows = [[c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])] for r in nested.find_all("tr")]
                nrows = [r for r in nrows if r]
                if len(nrows) >= 2 and _param_key(nrows[0][0] if nrows[0] else ""):
                    for lab, val in zip(nrows[0], nrows[1]):
                        k = _param_key(lab)
                        if k:
                            rec[k] = val
                else:
                    for r in nrows:
                        if len(r) >= 2 and _param_key(r[0]):
                            rec[_param_key(r[0])] = r[1]
            if rec.get("institute_name"):
                rows.append(rec)
    return rows


def load_html_folder():
    rows = []
    for f in sorted(RAW_HTML.glob("*.html")):
        html = f.read_text(encoding="utf-8", errors="ignore")
        m = re.match(r"(\d{4})__(.+)\.html", f.name)
        if m:
            year, page = int(m.group(1)), m.group(2)
        else:  # browser-saved page with a generic name: read "India Rankings 2024: Engineering" heading
            h = re.search(r"India Rankings\s*(\d{4})\s*:\s*([^<]+)", html)
            if not h:
                print("Skipped (cannot tell year/category):", f.name)
                continue
            year, page = int(h.group(1)), re.sub(r"[^A-Za-z]", "", h.group(2)) + "Ranking"
        rows += parse_html(html, year, page)
    return pd.DataFrame(rows)


COLUMN_ALIASES = {
    "institute_id": ["institute id", "institute_id", "id", "institute code", "code"],
    "institute_name": ["name", "institute name", "institute_name", "institution", "institution name", "college name"],
    "city": ["city", "town"],
    "state": ["state", "state/ut"],
    "score": ["score", "total score", "overall score", "nirf score"],
    "rank": ["rank", "nirf rank", "ranking", "overall rank"],
    "year": ["year", "ranking year", "ranking_year"],
    "category_original": ["category", "ranking category", "ranking_category", "discipline"],
    "TLR": ["tlr", "teaching, learning & resources", "teaching learning and resources"],
    "RPC": ["rpc", "rp", "research and professional practice", "research and professional practices"],
    "GO": ["go", "graduation outcomes"],
    "OI": ["oi", "outreach and inclusivity"],
    "PR": ["pr", "perception", "peer perception"],
}


def load_csv_folder():
    frames = []
    for f in sorted(RAW_CSV.glob("*.csv")):
        try:
            df = pd.read_csv(f, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin-1")
        norm = {c: re.sub(r"\s+", " ", str(c).strip().lower()) for c in df.columns}
        ren = {}
        for canon, aliases in COLUMN_ALIASES.items():
            for c, n in norm.items():
                if n in aliases and c not in ren and canon not in ren.values():
                    ren[c] = canon
                    break
        df = df.rename(columns=ren)
        if "year" not in df.columns:
            m = re.search(r"(20\d{2})", f.stem)
            if m:
                df["year"] = int(m.group(1))
        if "category_original" not in df.columns:
            stem = re.sub(r"(20\d{2})|[_\-]", " ", f.stem).strip()
            df["category_original"] = stem or "Unknown"
        df["source_page"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_all():
    frames = [f for f in (load_html_folder() if any(RAW_HTML.glob("*.html")) else pd.DataFrame(),
                          load_csv_folder() if any(RAW_CSV.glob("*.csv")) else pd.DataFrame()) if len(f)]
    if not frames:
        sys.exit("No input found: put pages in raw_html/ (via `scrape`) or CSVs in raw_csv/.")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(RAW_OUT, index=False)
    print(f"Wrote {RAW_OUT.name}: {len(raw):,} rows")
    print(raw.groupby(["year", "category_original"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("scrape", "all"):
        scrape()
    if cmd in ("parse", "all"):
        parse_all()