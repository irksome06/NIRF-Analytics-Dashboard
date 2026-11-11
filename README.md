# NIRF Rankings Dashboard (2017–2025)

Interactive Streamlit dashboard over the cleaned NIRF master dataset: 8 analysis sections (all 40 Annexure A
questions plus an India boundary map, a top-institutions leaderboard and a select-any-two-variables explorer), an institution explorer, a validated query engine, and an
"Ask NIRF" natural-language chat.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Run the tests (about 30 s): `python -m pytest -q tests`

Optional Gemini fallback for questions the rules cannot place:
```bash
export GEMINI_API_KEY="your-key"        # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
export GEMINI_MODEL="gemini-2.0-flash"  # optional; change if your key uses another model
```
Without a key the chatbot still works using the rule-based NLP.

## Layout (follows the architecture document)

```
nirf-dashboard/
  data/        nirf_master_final.csv                 M2  master dataset
               india_states.geojson                  simplified state boundaries (146 KB, offline map)
               cleaning/                             data cleaning code: 01_extract, 02_preprocess, 03_final_clean,
                                                     audit reports and cleaning log (see cleaning/README.md)
  engine/      data_loader.py  helpers.py            shared loader, filters, theme, rank groups
               registry.py     query_engine.py       question registry + validator + runner
  analysis/    trends.py       geo.py                M3 (Sections 1, 7)   M4 (Section 2 + map)
               stats.py        multivariate.py       M5 (Sections 3, 4, 6, 5, 8)
               institutions.py                       institution radar + comparison (Institutions page)
  nlp/         preprocess.py entities.py intent.py   M1 rule-based NLP
               gemini.py       chatbot.py            Gemini JSON fallback + pipeline
  app/         streamlit_app.py ui.py views/         M6 Streamlit multipage app
  tests/       336 tests: data checks, every question x 6 categories, validator, chatbot routing
```

**Shared contract.** Every analysis function is `fn(df, **filters) -> (result_df, plotly_figure)` and registers itself
once with `@question(...)` in `engine/registry.py`. Pages, validator and chatbot all read that registry, so adding a
question there makes it appear everywhere.

**Chat flow.** text → entity extraction (years, category, states, institution, parameter, counts) → intent
classifier → *(Gemini → JSON only, if unclear)* → `validate_query` → analysis function → chart + table.
The validator rejects unknown questions, fields, categories, states, institutions and out-of-range values, and
Gemini output goes through the same validator. Nothing generated is ever executed.

## Annexure A coverage

| Section | Questions | Page |
|---|---|---|
| 1 Trends | 1.1–1.5 | Trends |
| 2 Geography | 2.1–2.5, 2.6 map | Geography |
| 3 Univariate | 3.1–3.5 | Distributions |
| 4 Bivariate | 4.1–4.5, 4.6 any-two-variables scatter | Relationships |
| 5 Multivariate | 5.1–5.5 | Parameter profiles |
| 6 Distribution comparison | 6.1–6.5 | Statistical comparison |
| 7 Transitions | 7.1–7.5 | Rank mobility |
| 8 Outliers | 8.1–8.5 | Outliers and gaps |

Level 2: every analysis exposes its own filters (category, year or range, institution, state, parameter, thresholds)
generated from `PARAM_SPECS`, so unseen questions can be answered by changing selections.

## Method notes to be ready to explain

- **Ranks.** Exact ranks are used as published. Bands (e.g. 101–150) are grouped as bands; where one number is needed
  the midpoint is used and labelled as an estimate. The sidebar toggle "Exact published ranks only" controls this.
- **Rank groups** (Top 10, 11–50, 51–100, 101–200, 201+) are built from `rank_low`/`rank_high`, so exact ranks and bands
  share one scale in the transition analyses (Section 7).
- **Scores** are compared only within a `score_comparison_group`; trend charts draw one line per weighting scheme.
  5.4 and 8.3 refuse to compare scores across different schemes.
- **Top 100 / state counts** are per category (an institution can appear in several categories).
- **Outliers**: 8.1 uses residuals of RPC regressed on the overall score; 8.3 uses a median/MAD z-score of the year-on-year
  change after removing that year's median change; 8.5 compares rank standard deviation with peers of similar average rank.
  All thresholds are sliders.

## Known limits

- The map basemap loads from the internet; choose **Offline** map style in 2.6 if the venue network is unreliable.
- Fonts load from Google Fonts and fall back to system fonts offline.
- The Gemini path could not be exercised without a key; the code, prompt and validation are covered, the live call is not.
- Weighting schemes in the master file were inferred for most rows, and the published bands are marked "derived" in the data.
  Verify both against the source before you present them as official.

## Added in the merge
- **India boundary map** (question 2.6 default, also on Overview): offline choropleth with the best-ranked institution on hover. The bubble basemap and offline scatter remain as options.
- **Question 1.6 "Top institutions"**: category/year leaderboard, also reachable from chat ("top 10 engineering colleges in 2024").
- **Institutions page**: profile (rank path, parameter radar against the cohort and top-10 average) and compare up to five institutions.
- **Data and methods**: weighting-scheme table and a master-dataset download.
- 20 new tests (boundary map vs state table, leaderboard, routing guard-rails, radar and compare).

## Ask NIRF (natural-language questions)
Pipeline: `text → greeting/help/out-of-scope → spelling fix → entities → intent → follow-up resolution → (Gemini only if unclear) → validator → query engine`.

| Module | Role |
|---|---|
| `nlp/lexicon.py` | spelling correction against domain + institution vocabulary ("madrass" → "madras"); aliases like IISc are never "corrected" |
| `nlp/entities.py` | years (ranges, "since", "last 5 years", "2018-22"), category, states (short codes such as UP only when capitalised), top-N / bottom-N, parameters, institutions |
| `nlp/cues.py` | weighted, order-insensitive patterns per question (paraphrases, not just keyword phrases) |
| `nlp/semantic.py` | stem + synonym-class similarity as a soft signal for unseen wording |
| `nlp/intent.py` | combines phrases, cues, similarity and entity rules (one institution → 1.1 / 1.8, two → 1.7, leaderboard → 1.6) |
| `nlp/followup.py` | "what about engineering?", "and for 2022?", "show more", "include band estimates" |
| `nlp/chatbot.py` | orchestration; replies carry the interpretation, corrections, "did you mean" alternatives and next-step suggestions |
| `nlp/gemini.py` | optional fallback; few-shot prompt + conversation context; output is always validated, never executed |

New questions for chat and the dashboard: **1.6** top / bottom institutions (optionally one state), **1.7** compare two institutions, **1.8** institution snapshot.

### Evaluating it
```bash
python tests/nlq_eval.py -v      # accuracy by group, failures listed
```
`tests/nlq_cases.py` holds three question sets (paraphrases, typos, relative years, leaderboards, follow-ups, out-of-scope). The sets were written by the same
author as the rules, so treat the final numbers as optimistic: first-run accuracy on the two later sets, before any tuning, was 79% and 66% (intent).
Add real user questions to `CASES` as you collect them; `tests/test_nlq.py` fails if accuracy drops.

## Questions outside the 40 standard analyses (chart builder)

Two routes draw a chart for any combination of measure, grouping and filters:

* **Build a chart page** (`app/views/builder.py`): dropdowns for measure (score, TLR, RPC, GO, OI, PR, rank, count), aggregation,
  group by (year, state, category, city, institution), split by, chart type (bar, line, scatter, box, histogram, heatmap) and filters
  (category, years, states, institution family such as IIT or NIT, rank cut-off, top N, order). The chart redraws on every change.
* **Ask NIRF** (`nlp/custom_nlq.py`): when a question is not one of the 40, or is an explicit "measure by group" request, the rules
  read it into the same spec and draw it. Follow-ups such as *what about pharmacy?*, *now 2024*, *show it by year* modify the last chart.
  With a Gemini key, the model may also return a spec; it is validated exactly like a rules-built one.

Everything goes through `analysis/custom.py::build_chart`, which validates the spec against the data and an allowlist, then runs pandas.
No generated code is executed. Impossible requests return a message saying what to change (for example
*Dental has editions 2020-2025*). Tests: `tests/test_custom.py` (19 unseen questions, follow-ups, error messages, and a sweep over
every option combination).
