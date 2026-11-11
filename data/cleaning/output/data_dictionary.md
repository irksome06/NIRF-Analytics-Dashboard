# NIRF master dataset: data dictionary

Rows: 11,012 | Years: 2017-2025 | Categories: 16 | Institutions: 1,635

| Column | Type | Description | Missing % | Notes |
|---|---|---|---|---|
| institution_key | string | Stable internal ID for one institution across all years and categories (our own, not NIRF's). Built from state + normalised name + NIRF institute_id. | 0.0 | Join key for trends/mobility |
| institute_name | string | Standardised (canonical) institution name, taken from the broadest category (Overall > University > College > others). Same-name institutions in different states carry the state in brackets. | 0.0 | Use for all charts and filters |
| institute_id | string | Institute ID printed by NIRF (2019+ format IR-E-U-0456; older years use other formats). | 36.74 | Optional; many missing for band rows |
| institute_name_original | string | Name exactly as published that year, after whitespace/unicode cleanup. | 0.0 | Audit only |
| year | int | NIRF ranking year. | 0.0 | 2017-2025 |
| category | string | Standardised ranking category (Overall, Engineering, Pharmacy, ...). | 0.0 | Filter / facet. Never add up or average across categories: the same institution is listed in several categories in one year |
| category_original | string | Category label as published on the source page. | 0.0 | Keeps historical definitions |
| category_group | string | General (Overall, University, College, Research, ...) vs Discipline (Engineering, Pharmacy, ...). | 0.0 | Helps compare like with like |
| city | string | Cleaned city (no 'District', PIN/sector numbers or trailing state; Gurgaon->Gurugram, Calcutta->Kolkata, ...). | 0.0 | Prefer state for maps |
| city_original | string | City as published. | 0.09 | Audit only |
| state | string | Standardised state/UT name (e.g. Orissa -> Odisha). It is the location of that year's listing. | 0.0 | Use for geospatial analysis (state-level maps) |
| state_code | string | ISO 3166-2:IN two-letter state/UT code (TN, KA, DL, ...). | 0.0 | Join key for state GeoJSON / choropleth |
| state_original | string | State as published. | 0.0 | Audit only |
| rank | float | Exact NIRF rank (1 = best). Null when rank_status is band, not_published or withdrawn. Ties share a rank. | 36.84 | Never imputed except rank_status = inferred_from_score |
| rank_value | float | rank if exact, otherwise rank_midpoint of the band. Use when every institution needs a number (mobility, improvement, volatility). | 0.1 | Banded rows are approximate (+/- half a band) |
| rank_status | string | exact = published rank; band = listed only in a rank band; not_published = listed with no rank (Innovation 2023); withdrawn = 'Rank Withdrawn'; inferred_from_score = rank missing in the source but the score fits exactly one free slot in the published sequence. | 0.0 | Filter on this instead of testing rank for null |
| rank_original | string | Rank text exactly as published (e.g. '21A', 'Rank Withdrawn'). | 36.83 | Audit only |
| rank_low | float | Lower bound of the rank (equals rank for exact ranks; start of the band for banded rows). | 0.1 |  |
| rank_high | float | Upper bound of the rank (equals rank for exact ranks; end of the band for banded rows). | 0.1 |  |
| is_rank_band | bool | True if the institution is listed only in a rank band (no exact rank, score or parameters). | 0.0 |  |
| rank_band_label | string | Band as 'low-high', e.g. '101-150' or '201-250'. | 63.26 | DERIVED, see rank_band_source |
| rank_band_source | string | 'derived': the band starts after the last exact rank of that year+category; pages '...150' then '...200'; width = institutions on the page rounded to 25. | 63.26 | Verify a few bands on nirfindia.org |
| rank_midpoint | float | (rank_low + rank_high)/2. Filled for exact ranks and for bands. | 0.1 |  |
| score | float | NIRF overall score (0-100) for that category and year. Null for band rows and a few small categories. | 37.13 |  |
| TLR | float | Teaching, Learning & Resources score (0-100). | 39.4 | Parameter 1 |
| RPC | float | Research and Professional Practice score (0-100). Called 'RP' in older years. | 39.4 | Parameter 2 |
| GO | float | Graduation Outcomes score (0-100). | 39.4 | Parameter 3 |
| OI | float | Outreach and Inclusivity score (0-100). | 39.4 | Parameter 4 |
| PR | float | Perception score (0-100). | 39.4 | Parameter 5 |
| weight_scheme | string | Weights (%) of TLR/RPC/GO/OI/PR in the score for that category and year, estimated from the data (R2 = 1). E.g. 30/30/20/10/10. | 3.11 | Scores are only strictly comparable under the same scheme (see score_weights_estimated.csv) |
| is_top10 | bool | rank <= 10 within its category and year. | 0.0 | Derived |
| is_top50 | bool | rank <= 50 within its category and year. | 0.0 | Derived |
| is_top100 | bool | rank <= 100 (banded rows count only if the whole band lies within 100). | 0.0 | Derived |
| is_new_entry | bool | First year the institution appears in that category (excluding the category's first year). | 0.0 | Derived; for 'newly appearing' questions |
| first_year_in_category | int | First year the institution appears in that category. | 0.0 | Derived |
| category_first_year | int | First year the category exists in the data (e.g. Dental, Innovation start later than 2017). | 0.0 | Derived; handles category changes |
| categories_in_year | int | Number of categories published that year (6 in 2017, more later). | 0.0 | Derived |
| source_page | string | Source file / page name the row was taken from. | 0.0 | Lineage |

## Notes for analysis

- Do not add up or average across categories: an institution has about 1.5 rows per year (Overall, University, Engineering, ...). Always filter or group by category.
- Band rows (about 37%) have no exact rank, score or parameters. Use rank_value for rank-movement questions and say that banded ranks are approximate.
- Band ranges are derived from the last exact rank of each category-year, not read from the site: check a few against nirfindia.org.
- The score weights changed by category and year (e.g. College 2017 vs 2018+). Compare scores across years only within one weight_scheme, or note the difference.
- An institution can appear in two states only when NIRF's own institute_id links the listings (e.g. Amrita School of Pharmacy, Kochi = Amrita Vishwa Vidyapeetham). Use the row's state for state-level maps.
- Research, Innovation, Open University and Skill University use different parameters/scoring: TLR..PR are empty for them.

## Score weights by category and year (TLR/RPC/GO/OI/PR, %)

Estimated from the data by least squares; R2 = 1.000 means exact.

| Category | Weights | Years |
|---|---|---|
| Agriculture | 30/30/20/10/10 | 2023, 2024, 2025 |
| Architecture | 40/20/20/10/10 | 2020, 2021, 2022, 2023, 2024, 2025 |
| College | 30/20/25/15/10 | 2017 |
| College | 40/15/25/10/10 | 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 |
| Dental | 30/30/20/10/10 | 2020, 2021, 2022, 2023, 2024, 2025 |
| Engineering | 30/30/20/10/10 | 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 |
| Law | 40/15/25/10/10 | 2020, 2021, 2022, 2023, 2024, 2025 |
| Management | 30/30/20/10/10 | 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 |
| Medical | 30/30/20/10/10 | 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 |
| Overall | 30/30/20/10/10 | 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 |
| Pharmacy | 30/30/20/10/10 | 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 |
| State Public University | 30/30/20/10/10 | 2024, 2025 |
| University | 30/30/20/10/10 | 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025 |