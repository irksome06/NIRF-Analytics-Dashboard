# Data cleaning pipeline

Code that turns the NIRF pages into `data/nirf_master_final.csv`, the file the dashboard loads.

```
nirfindia.org pages (raw_html/)
   │  01_extract.py        scrape / parse HTML  ->  raw_extracted.csv   (19,742 raw rows)
   ▼
   │  02_preprocess.py     clean, standardise, de-duplicate  ->  output/nirf_master.csv
   ▼                       (+ output/data_dictionary.*, output/audit/*)
   │  03_final_clean.py    verified fixes only  ->  ../nirf_master_final.csv  (+ cleaning_log.csv)
   ▼
dashboard
```

## Re-run

```bash
pip install pandas numpy pyarrow beautifulsoup4 lxml requests
cd data/cleaning
python 01_extract.py parse     # needs raw_html/ (not shipped, ~25 MB; `python 01_extract.py scrape` re-downloads it)
python 02_preprocess.py        # raw_extracted.csv is included, so you can start here
python 03_final_clean.py
```
Running steps 2 and 3 from the shipped `raw_extracted.csv` reproduces `nirf_master_final.csv` exactly.

## What each step does

**Step 2: `02_preprocess.py`** (11,012 rows out; 8,730 duplicates removed)
- standardises state and category names, parses ranks into exact / band / not published / withdrawn
- sets out-of-range scores and parameters (outside 0-100) to null; none were found in this run
- builds one `institution_key` per institution, merging spelling variants and keeping same-name institutions in different cities apart
- fills missing state/city from the same institution's other years only; ranks and scores are never inferred
- estimates the weighting scheme per category and year (`output/score_weights_estimated.csv`)
- optional `manual_aliases.csv` (`raw_name,canonical_name`) overrides institution-name matching

**Step 3: `03_final_clean.py`**: fixes provable from the data itself, each logged in `cleaning_log.csv`

| step | rows | change |
|---|---|---|
| withdrawn_fix | 1 | Dental 2025 rank 18 marked withdrawn (raw score says "Rank Withdrawn*") |
| name_text | 11 | `?` and doubled apostrophes, stray trailing digits |
| merge_keys | 19 | institutions split by punctuation/spacing variants re-joined |
| city | 38 | placeholder or state-as-city values, "locality, district" strings |
| weight_scheme | 50 | Architecture/Law 2018-19 filled where the recomputed score matches within 0.05 |

It then recomputes the derived columns (top-10/50/100 flags, new-entry flags) and asserts the integrity rules
(no duplicate institution-year-category rows, one name per key, scores within 0-100).

## Audit files (`output/`)
`data_dictionary.md/.csv` column definitions · `audit/data_quality_report.csv` · `audit/cleaning_summary.csv` ·
`audit/institution_aliases.csv` raw name to canonical name · `audit/possible_duplicates_review.csv` near-matches kept separate for manual review ·
`scrape_log.csv` download log.
