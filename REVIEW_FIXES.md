# Fixes applied in code review

1. **nlp/chatbot.py** – `parse_custom` was imported but never called on the rules path, so the chart builder was reachable only
   through Gemini. Wired in with the "inherit / strong / weak" protocol described in `nlp/custom_nlq.py`; the parser is given the
   previous query only when it was itself a custom chart. Fixes the 20 failing tests in `tests/test_custom.py`.
2. *(removed on request – the Rank Simulator page no longer exists)* **app/views/simulator.py** – (a) used a hand-typed "official" weight table that disagrees with the dataset's own `weight_scheme`
   for Law, Agriculture, Architecture and College 2017 (errors of 2-5 points); now reads the scheme from the data.
   (b) recomputed the score from scratch and counted the institution against itself, so with all sliders at zero 42-54% of
   institutions (92-95% in Law/Agriculture) showed a rank change. Now `sim_score = published score + weighted changes` and the
   institution is excluded from its own rank count.
3. **analysis/geo.py, nlp/insights.py** – "national share" in the 2.1 insight divided by the sum of the *displayed* states (75),
   giving 22.7% for Tamil Nadu instead of 17.0%, and changing with "how many to show". Now uses the full top-100 total.
4. **tests/test_review_fixes.py** – regression tests for the above. Full suite: 449 passed before these tests were added.
5. **Truncated institution names** (KPI cards showed "Indian Institute of", scatter labels showed "Indian Institute of Techno").
   Names were cut at a fixed character count in ~20 places. Added `short_name()` in `engine/helpers.py`:
   - KPI cards, tables, bar axes and hover text now show the full name (bars trim only beyond 60 characters, at a word boundary).
   - On-chart point labels use readable abbreviations ("IIT Madras", "NIT Rourkela", "IIT (BHU) Varanasi"), alternate their
     position to reduce overlap, and cut only at word boundaries; hover always shows the full name.
   - Tests check every institution name in the data, and that KPI values are never a truncated prefix of a longer name.

6. **Rank Simulator removed** at the team's request: page file, menu entry, import and its tests deleted. The chatbot still declines
   prediction questions ("I can only analyse the published NIRF data…").
