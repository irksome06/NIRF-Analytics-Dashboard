"""Ask NIRF chatbot pipeline:
   text -> special cases (greeting / help / out of scope) -> spelling fix -> entities -> intent -> follow-up resolution
        -> (Gemini only if still unclear) -> validator -> query engine -> reply with interpretation, alternatives and next steps."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pandas as pd

from analysis.custom import CustomChartError, build_chart, describe
from engine.data_loader import year_range
from engine.query_engine import AmbiguousInstitution, QueryResult, QueryValidationError, default_params, run_query, validate_query
from engine.registry import QUESTIONS
from nlp.custom_nlq import parse_custom
from nlp.entities import extract
from nlp.followup import resolve
from nlp.gemini import ask_gemini, get_api_key
from nlp.insights import generate_insights
from nlp.intent import THRESHOLD, classify
from nlp.lexicon import spell_fix
from nlp.preprocess import normalise

EXAMPLES = [
    "How has IIT Madras ranking changed from 2017 to 2025?",
    "Top 10 engineering colleges in 2024",
    "Compare IIT Bombay and IIT Delhi",
    "Which engineering institutions improved the most between 2018 and 2025?",
    "Which states have the most institutions in the top 100 in 2025?",
    "Compare Karnataka and Tamil Nadu in engineering",
    "Scatter plot of TLR vs RPC for universities in 2024",
    "Which institutions entered or exited the top 10?",
]
TOPICS = {
    "Custom charts": ["Average TLR of engineering colleges by state in 2022", "How many institutions per city are ranked in 2025?", "Box plot of scores by category in 2024", "Average GO of IITs over the years"],
    "Rankings": ["Top 10 engineering colleges in 2024", "Who climbed the most places since 2020?", "How did NIT Warangal do in 2022?"],
    "Compare": ["Compare IIT Bombay and IIT Delhi", "Karnataka vs Tamil Nadu for universities", "Compare the top 5 institutions across parameters"],
    "States and map": ["Which states have the most institutions in the top 100?", "Show the map of India", "Best colleges in Kerala in 2024"],
    "Scores": ["Distribution of scores in 2023", "Which parameter correlates most with the overall score?", "Scatter of TLR vs RPC for universities"],
    "Movement": ["Who entered the top 10 this year?", "Which institutions have unusually volatile ranks?", "Largest rank jumps between 2023 and 2025"],
}
CAPABILITIES = ("You can ask about rankings and trends (a leaderboard, an institution's rank path or yearly profile, biggest risers and fallers), "
                "comparisons (two institutions, two states), states and the map, score distributions and correlations, parameter profiles, "
                "and rank movement or outliers. Name a category, year or institution if you want, and follow up with things like "
                "*what about engineering?* or *show more*.")
FALLBACK = ("I couldn't match that to one of the analyses. " + CAPABILITIES)
NEXT = {
    "1.2": ["and the biggest decliners?", "include band estimates", "what about engineering?"],
    "1.3": ["and the biggest improvers?", "include band estimates", "what about engineering?"],
    "1.6": ["show more", "bottom 10", "what about 2022?"],
    "1.1": ["compare it with IIT Bombay", "how did it do in 2023?", "include band estimates"],
    "1.7": ["include band estimates", "since 2020", "profile of the first one"],
    "1.8": ["trend over the years", "what about 2020?", "compare with IIT Delhi"],
    "2.1": ["show the map of india", "what about engineering?", "which states gained representation?"],
    "2.4": ["what about universities?", "which states produced the most new institutions?", "show the map of india"],
}
CATEGORY_HOPS = ["Engineering", "University", "College", "Pharmacy", "Management"]


@dataclass
class ChatReply:
    ok: bool
    message: str
    result: QueryResult | None = None
    query: dict = field(default_factory=dict)
    source: str = "rules"
    assumed: dict = field(default_factory=dict)
    resolved: dict = field(default_factory=dict)          # validated {"question", **params}: store as context for the next turn
    corrections: dict = field(default_factory=dict)       # {"madrass": "madras"}
    followup: bool = False
    alternatives: list = field(default_factory=list)      # [(label, prompt)] other readings of an unclear question
    suggestions: list = field(default_factory=list)       # [prompt] sensible next questions
    insights: list[str] = field(default_factory=list)     # Bullet points summarizing key takeaways
    metrics: list[dict] = field(default_factory=list)     # KPI cards: [{"label": "...", "value": "..."}]
    disambiguation: list[str] = field(default_factory=list) # Candidate institutions if ambiguous
    latency_ms: float = 0.0


def _infer_category(df: pd.DataFrame, institution_name: str) -> str | None:
    cats = df.loc[df["institute_name"].eq(institution_name), "category"].value_counts()
    if cats.empty:
        return None
    return "Overall" if "Overall" in cats.index else cats.index[0]


def _describe(qid: str, params: dict, df: pd.DataFrame, assumed: set[str]) -> str:
    def tag(keys, text):
        return text + (" (default)" if all(k in assumed for k in keys) else "")
    bits = []
    if "category" in params:
        bits.append(tag(["category"], params["category"]))
    if "year" in params:
        bits.append(tag(["year"], str(params["year"])))
    if "year_from" in params and "year_to" in params:
        bits.append(tag(["year_from", "year_to"], f"{params['year_from']}–{params['year_to']}"))
    if "year_a" in params and "year_b" in params:
        bits.append(tag(["year_a", "year_b"], f"{params['year_a']} → {params['year_b']}"))
    for k in ("state_a", "state_b", "state_filter", "parameter"):
        if params.get(k):
            bits.append(tag([k], str(params[k])))
    if params.get("top_n") and "top_n" in QUESTIONS[qid].params:
        bits.append(f"{'bottom' if params.get('bottom') else 'top'} {params['top_n']}")
    for k in ("institution", "institution_b"):
        if params.get(k):
            row = df.loc[df["institution_key"].eq(params[k]), "institute_name"]
            if not row.empty:
                bits.append(row.iloc[0])
    return f"**{QUESTIONS[qid].title}**  \n_Filters: {', '.join(bits)}_"


def _custom_reply(df: pd.DataFrame, spec: dict, fixes: dict, source: str = "rules", followup: bool = False) -> ChatReply:
    """Answer with the generic chart builder (a question that is not one of the 40 registered analyses)."""
    extra = list(spec.pop("_notes", []))
    try:
        data, fig, notes, clean = build_chart(df, spec)
    except CustomChartError as e:
        return ChatReply(False, str(e), query={"custom": spec}, source=source, corrections=fixes)
    except Exception:
        return ChatReply(False, "That chart could not be drawn with these filters. Try another category or year range.", query={"custom": spec}, source=source, corrections=fixes)
    title = describe(clean)
    msg = f"**{title[0].upper() + title[1:]}**  \n_Drawn with the chart builder, since this is not one of the standard analyses. Chart type: {clean['chart']}._"
    for n in extra + notes[:2]:
        msg += f"  \n_{n}_"
    if followup:
        msg += "  \n_Continuing from your previous question._"
    if fixes:
        msg += "  \n_Read as: " + ", ".join(f"“{a}” → “{b}”" for a, b in fixes.items()) + "_"
    if source == "gemini":
        msg += "  \n_Interpreted with Gemini, then validated._"
    res = QueryResult(qid="custom", title=title, params=clean, data=data, figure=fig, notes=notes)
    nxt = ["what about engineering?", "now 2020", "show it as a box plot" if clean["chart"] != "box" and clean["metric"] not in ("count",) and clean.get("group_by") else "show it by year"]
    bullets, kpis = generate_insights("custom", clean, data)
    return ChatReply(True, msg, res, {"custom": clean}, source, {}, {"question": "custom", **clean}, fixes, followup, [], nxt, insights=bullets, metrics=kpis)


def _special(raw: str, low: str, df: pd.DataFrame) -> ChatReply | None:
    y0, y1 = year_range(df)
    if re.match(r"^(?:hi+|hello|hey+|namaste|good (?:morning|afternoon|evening))\b[ !.]*$", low):
        return ChatReply(False, "Hello! " + CAPABILITIES)
    if re.match(r"^(?:help|what can you do|how (?:do|does) (?:this|it) work|examples?|what can i ask|options|capabilities)\b", low):
        return ChatReply(False, CAPABILITIES)
    if re.match(r"^(?:thanks|thank you|thx|ok|okay|cool|great|nice|got it)\b[ !.]*$", low):
        return ChatReply(False, "Glad to help. Ask another question, or try a follow-up like *what about engineering?*")
    if re.search(r"\b(?:predict\w*|forecast\w*|will be|next year|in the future|projection)\b", low):
        return ChatReply(False, f"I can only analyse the published NIRF data for {y0}–{y1}, so I can't predict future rankings. "
                                "I can show the recent trend instead, for example: *average score trend over the last 5 years*.")
    out = [int(y) for y in re.findall(r"\b(20[0-9]{2})\b", low) if not y0 <= int(y) <= y1]
    if out:
        return ChatReply(False, f"This dashboard covers NIRF editions {y0}–{y1}; {out[0]} is outside that range. Try a year inside it.")
    if re.search(r"\b(?:weather|joke|stocks?|cricket|movies?|recipes?|news|capital of|world cup|election|bitcoin|football|song)\b", low):
        return ChatReply(False, FALLBACK)
    return None


def _prompt_for(qid: str, ents: dict) -> str:
    t = QUESTIONS[qid].title.rstrip("?")
    if ents.get("category"):
        t += f" in {ents['category']}"
    if ents.get("year"):
        t += f" in {ents['year']}"
    return t


def _next_steps(qid: str, params: dict) -> list[str]:
    out = list(NEXT.get(qid, ["what about engineering?", "now 2020", "show the map of india"]))
    cur = params.get("category")
    hop = next((c for c in CATEGORY_HOPS if c != cur), None)
    return [(f"what about {hop.lower()}?" if s == "what about engineering?" and hop else s) for s in out if not (s.startswith("what about") and cur == "Engineering" and "engineering" in s)][:3]


def answer(df: pd.DataFrame, text: str, context: dict | None = None, use_gemini: bool = True, pool: pd.DataFrame | None = None) -> ChatReply:
    t0 = time.perf_counter()
    raw = (text or "").strip()
    if not raw:
        return ChatReply(False, "Type a question about the NIRF rankings, for example: *" + EXAMPLES[0] + "*")
    special = _special(raw, normalise(raw), df)
    if special:
        return special
    fixed, fixes = spell_fix(raw, df)
    ents = extract(fixed, df, pool, raw=raw)
    qid, score, ranked = classify(fixed, ents)
    last = (context or {}).get("last_query")
    last_custom = last if last and last.get("question") == "custom" else None
    spec, strength = parse_custom(fixed, df, ents, raw=raw, last=last_custom)
    if spec and strength == "inherit":            # follow-up of the previous custom chart
        return _custom_reply(df, spec, fixes, followup=True)
    if spec and strength == "strong":             # clearly not one of the registered analyses
        return _custom_reply(df, spec, fixes)
    qid, ents, followup = resolve(fixed, ents, qid, score, last)
    source = "rules"
    query: dict | None = None
    # After special handling, if Gemini is forced, try it first
    if use_gemini and get_api_key():
        g = ask_gemini(raw, df, context=last)
        if g:
            # If Gemini returns a custom chart spec, handle it directly
            if isinstance(g.get("custom"), dict):
                return _custom_reply(df, dict(g["custom"]), fixes, source="gemini")
            # Otherwise Gemini returned a structured query dict
            query = g
            source = "gemini"
    # Existing rule‑based path (only if we didn’t get a query from Gemini)
    if query is None:
        # Determine which registry question (qid) to use based on entities
        if qid is None and ents.get("institution") and len(fixed.split()) <= 7:
            qid = "1.1"  # a bare institution name defaults to its rank trend
        if qid is not None:
            query = {"question": qid, **ents}
            if qid == "1.6" and ents.get("state") and not ents.get("state_filter"):
                query["state_filter"] = ents["state"]
        elif use_gemini and get_api_key():
            # Fallback to Gemini only if we didn’t already try it above (unlikely)
            g = ask_gemini(raw, df, context=last)
            if g and isinstance(g.get("custom"), dict):
                return _custom_reply(df, dict(g["custom"]), fixes, source="gemini")
            if g:
                query, source = g, "gemini"
    if query is None and spec:                    # weak custom reading, used only when nothing else matched
        return _custom_reply(df, spec, fixes)
    if query is None:
        return ChatReply(False, FALLBACK, corrections=fixes, latency_ms=(time.perf_counter() - t0) * 1000)

    if "category" not in query:
        if query.get("institution") and "institution" in QUESTIONS.get(str(query.get("question")), QUESTIONS["1.1"]).params:
            inferred = _infer_category(df, query["institution"]) if query["institution"] in set(df["institute_name"]) else None
            query["category"] = inferred or (context or {}).get("category", "Overall")
        elif context and context.get("category"):
            query["category"] = context["category"]
    try:
        qid_final, params = validate_query(df, query)
        defaults = default_params(df, qid_final)
        assumed = {k: v for k, v in params.items() if k not in query and k in defaults}
        result = run_query(df, {"question": qid_final, **{k: v for k, v in query.items() if k != "question"}})
    except AmbiguousInstitution as e:
        candidates = getattr(e, "candidates", [])
        return ChatReply(False, str(e), query=query, source=source, corrections=fixes, disambiguation=candidates[:8], latency_ms=(time.perf_counter() - t0) * 1000)
    except QueryValidationError as e:
        return ChatReply(False, f"{e}", query=query, source=source, corrections=fixes, latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception:  # analysis failed unexpectedly: never crash the chat
        return ChatReply(False, "That analysis could not be completed with these filters. Try changing the category or years.", query=query, source=source, corrections=fixes, latency_ms=(time.perf_counter() - t0) * 1000)
    msg = _describe(result.qid, result.params, df, set(assumed))
    if followup:
        msg += "  \n_Continuing from your previous question._"
    if fixes:
        msg += "  \n_Read as: " + ", ".join(f"“{a}” → “{b}”" for a, b in fixes.items()) + "_"
    if source == "gemini":
        msg += "  \n_Interpreted with Gemini, then validated._"
    alts = []
    if source == "rules" and len(ranked) > 1 and score < THRESHOLD + 4:
        alts = [(QUESTIONS[q].title, _prompt_for(q, ents)) for q, s in ranked[1:3] if q != result.qid and s >= 0.6 * score]
    bullets, kpis = generate_insights(result.qid, result.params, result.data)
    latency = (time.perf_counter() - t0) * 1000
    return ChatReply(True, msg, result, query, source, assumed, {"question": result.qid, **result.params}, fixes, followup, alts, _next_steps(result.qid, result.params), insights=bullets, metrics=kpis, latency_ms=latency)
