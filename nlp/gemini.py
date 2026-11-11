"""M1/M6 - Optional Gemini fallback. Gemini ONLY translates a question into a structured query (JSON).

It never writes or runs code: whatever it returns is parsed as JSON and then passed through the
same validator as every other query, which rejects unknown questions, fields or values.
Set GEMINI_API_KEY (environment variable or Streamlit secret). GEMINI_MODEL is configurable.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

import pandas as pd

from engine.data_loader import PARAMS, category_options, state_options
from engine.registry import PARAM_SPECS, QUESTIONS

DEFAULT_MODEL = "gemini-2.0-flash"


def get_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")  # type: ignore[no-any-return]
    except Exception:
        return None


FEW_SHOT = """Examples:
Q: how did iit madras do between 2019 and 2023 -> {"question": "1.1", "institution": "IIT Madras", "year_from": 2019, "year_to": 2023}
Q: top 10 pharmacy colleges in 2024 -> {"question": "1.6", "category": "Pharmacy", "year": 2024, "top_n": 10}
Q: compare iit bombay and iit delhi -> {"question": "1.7", "institution": "IIT Bombay", "institution_b": "IIT Delhi"}
Q: which states have the most top 100 universities -> {"question": "2.1", "category": "University"}
Q: plot oi against go for engineering in 2023 -> {"question": "4.6", "x": "OI", "y": "GO", "category": "Engineering", "year": 2023}
Q: who will win the world cup -> {"question": null}
"""


def build_prompt(text: str, df: pd.DataFrame, context: dict | None = None) -> str:
    ctx = f"The previous question in this conversation was: {json.dumps({k: v for k, v in context.items() if v is not None})}. If the new question is a follow-up, keep its fields unless the user changes them.\n" if context else ""
    catalogue = "\n".join(f'- "{q.qid}": {q.title}  [fields: {", ".join(q.params)}]' for q in QUESTIONS.values())
    return f"""You convert a user's question about NIRF rankings (2017-2025) into ONE JSON object. Output JSON only.
Choose "question" from this list:
{catalogue}

Allowed values:
- category: {category_options(df)}
- state / state_a / state_b: {state_options(df)}
- parameter / x / y: {PARAMS} (x and y may also be "score" or "rank_value")
- years: integers 2017-2025 (year, year_from, year_to, year_a, year_b)
- institution: the institution name as the user wrote it
- top_n / top_k: integer 3-30 ; exact_only / trendline / standardize: true or false
Use only fields listed for the chosen question. Omit anything the user did not state.
If no listed question fits but the user wants a chart of a measure (score, TLR, RPC, GO, OI, PR, rank or count) grouped by year, state, category, city or institution, return
{{"custom": {{"metric": ..., "agg": "mean|median|min|max|sum|count", "group_by": ..., "split_by": ..., "chart": "auto|bar|line|scatter|box|histogram|heatmap",
"metric_y": ..., "category": ..., "categories": [...], "states": [...], "name_contains": "IIT", "year": ..., "year_from": ..., "year_to": ..., "rank_max": ..., "top_n": ..., "sort": "asc|desc"}}}}
using only the fields needed (all values from the allowed lists above).
If the question cannot be answered at all, return {{"question": null}}.

{FEW_SHOT}
{ctx}User question: {text}
JSON:"""


def ask_gemini(text: str, df: pd.DataFrame, api_key: str | None = None, model: str | None = None, timeout: int = 15, context: dict | None = None) -> dict | None:
    key = api_key or get_api_key()
    if not key:
        return None
    model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": build_prompt(text, df, context)}]}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
        raw = payload["candidates"][0]["content"]["parts"][0]["text"]
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) and (obj.get("question") or isinstance(obj.get("custom"), dict)) else None
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError, OSError):
        return None
