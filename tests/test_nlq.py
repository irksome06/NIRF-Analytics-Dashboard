"""NLQ regression tests: held-out question sets, self-consistency, spelling, follow-ups, safety."""
import json
from unittest import mock

import pytest

import engine.query_engine  # noqa: F401  (registers every question)
from engine.registry import QUESTIONS
from nlp.chatbot import answer
from nlp.gemini import ask_gemini
from nlp.lexicon import spell_fix
from tests.nlq_cases import CASES, CASES_B, CASES_C
from tests.nlq_eval import check


def _score(df, cases):
    ok = sum(check(df, c)[1] for c in cases)
    return ok / len(cases)


def test_accuracy_tuned_set(df):
    assert _score(df, CASES) >= 0.97


def test_accuracy_second_set(df):
    assert _score(df, CASES_B) >= 0.95


def test_accuracy_third_set(df):
    assert _score(df, CASES_C) >= 0.93


@pytest.mark.parametrize("qid", sorted(QUESTIONS, key=lambda s: [int(x) for x in s.split(".")]))
def test_every_question_title_routes_to_itself(df, qid):
    r = answer(df, QUESTIONS[qid].title, use_gemini=False)
    if qid in ("1.1", "1.7", "1.8"):      # these need an institution
        assert not r.ok or r.result.qid == qid
    else:
        assert r.ok and r.result.qid == qid, (QUESTIONS[qid].title, r.message[:80])


def test_spelling_correction_keeps_aliases_and_names(df):
    fixed, fixes = spell_fix("how has iisc and iit madrass done", df)
    assert "iisc" in fixed and fixes == {"madrass": "madras"}


def test_up_is_not_uttar_pradesh(df):
    r = answer(df, "how many institutions went up, went down or stayed put between 2022 and 2024", use_gemini=False)
    assert r.ok and r.result.qid == "7.5" and "state" not in r.query
    r = answer(df, "best colleges in UP", use_gemini=False)
    assert r.result.params.get("state_filter") == "Uttar Pradesh"


def test_follow_up_chain(df):
    r1 = answer(df, "top 10 engineering colleges in 2024", use_gemini=False)
    r2 = answer(df, "show more", context={"last_query": r1.resolved}, use_gemini=False)
    assert r2.followup and r2.result.qid == "1.6" and r2.result.params["top_n"] == 20 and r2.result.params["category"] == "Engineering"
    r3 = answer(df, "what about pharmacy?", context={"last_query": r2.resolved}, use_gemini=False)
    assert r3.result.params["category"] == "Pharmacy" and r3.result.params["top_n"] == 20
    r4 = answer(df, "now 2022", context={"last_query": r3.resolved}, use_gemini=False)
    assert r4.result.params["year"] == 2022


def test_followup_does_not_hijack_a_new_question(df):
    last = answer(df, "top 10 engineering colleges in 2024", use_gemini=False).resolved
    r = answer(df, "which states have the most institutions in the top 100", context={"last_query": last}, use_gemini=False)
    assert r.result.qid == "2.1" and not r.followup


def test_replies_carry_suggestions_and_interpretation(df):
    r = answer(df, "top 10 engineering colleges in 2024", use_gemini=False)
    assert r.ok and r.suggestions and "Filters" in r.message and r.resolved["question"] == "1.6"


def test_corrections_are_reported(df):
    r = answer(df, "enginering colleges that improved the most", use_gemini=False)
    assert r.ok and r.corrections == {"enginering": "engineering"} and "engineering" in r.message.lower()


@pytest.mark.parametrize("text,needle", [("hi", "You can ask"), ("help", "You can ask"), ("predict the 2026 rankings", "can't predict"),
                                         ("rankings for 2031", "outside that range"), ("thanks", "Glad")])
def test_special_messages(df, text, needle):
    r = answer(df, text, use_gemini=False)
    assert not r.ok and needle in r.message


def test_gemini_output_is_always_validated(df):
    """A malicious or malformed Gemini reply must be rejected by the validator, never executed."""
    payload = {"candidates": [{"content": {"parts": [{"text": json.dumps({"question": "1.6", "category": "Nonexistent; DROP TABLE", "year": 2024})}]}}]}

    class Resp:
        def read(self):
            return json.dumps(payload).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with mock.patch("nlp.gemini.get_api_key", return_value="k"), mock.patch("nlp.chatbot.get_api_key", return_value="k"), \
            mock.patch("urllib.request.urlopen", return_value=Resp()):
        r = answer(df, "zzz qqq unmatched phrase", use_gemini=True)
    assert not r.ok and r.source == "gemini" and "Unknown category" in r.message


def test_gemini_prompt_contains_context_and_examples(df):
    from nlp.gemini import build_prompt
    p = build_prompt("and for 2022?", df, {"question": "1.6", "category": "Engineering"})
    assert "previous question" in p and "Examples:" in p and '"1.7"' in p
