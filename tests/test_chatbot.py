import pytest

from nlp.chatbot import answer

CASES = [
    ("How has IIT Madras ranking changed from 2017 to 2025?", "1.1"),
    ("show anna university trend", "1.1"),
    ("Which engineering institutions improved the most between 2018 and 2025?", "1.2"),
    ("which colleges dropped the most in rank", "1.3"),
    ("how has the average score changed over time", "1.4"),
    ("which institutions have consistently stayed in the top 10", "1.5"),
    ("Which states have the most institutions in the top 100 in 2025?", "2.1"),
    ("how is the geographical distribution of top institutions changing", "2.2"),
    ("which states gained representation in the top 100", "2.3"),
    ("Compare Karnataka and Tamil Nadu in engineering", "2.4"),
    ("which states produced the most new institutions", "2.5"),
    ("show me the map of india", "2.6"),
    ("what is the distribution of scores in 2022", "3.1"),
    ("average, median, minimum and maximum score in 2024", "3.3"),
    ("what percentage of institutions fall in each score quartile", "3.4"),
    ("how does the spread of scores differ across categories", "3.5"),
    ("Which parameter has the strongest correlation with the overall score?", "4.3"),
    ("scatter plot of TLR vs RPC for universities in 2024", "4.6"),
    ("has the correlation between research and overall score changed over time", "4.5"),
    ("top 5 institutions compared across parameters in 2024", "5.1"),
    ("institutions with high research but low teaching", "5.2"),
    ("which institutions have high teaching but low perception", "5.3"),
    ("which institutions improved their overall score despite declining parameters", "5.4"),
    ("which institutions have the largest gap between strongest and weakest parameters", "5.5"),
    ("how do the top 10 compare with the rest of the institutions", "6.1"),
    ("are scores more dispersed in some years than others", "6.2"),
    ("which parameter has the greatest variability", "6.3"),
    ("median and iqr over time", "6.4"),
    ("score distribution across states", "6.5"),
    ("how many moved from the 51-100 band into the top 50", "7.1"),
    ("which institutions entered or exited the top 10", "7.2"),
    ("percentage of institutions that stayed in the same rank band", "7.3"),
    ("who had the largest upward and downward rank movements", "7.4"),
    ("how many institutions moved up, down or stayed each year", "7.5"),
    ("which institutions have unusually high research scores relative to overall", "8.1"),
    ("who performs substantially better in research than teaching", "8.2"),
    ("which institutions had an unusually large year to year score change", "8.3"),
    ("strong parameter score but poor overall rank", "8.4"),
    ("which institutions have unusually high ranking volatility", "8.5"),
]


@pytest.mark.parametrize("text,qid", CASES)
def test_routing(df, text, qid):
    r = answer(df, text, use_gemini=False)
    assert r.ok, r.message
    assert r.result.qid == qid


def test_nonsense_gets_friendly_fallback(df):
    r = answer(df, "what is the weather today", use_gemini=False)
    assert not r.ok and "couldn't match" in r.message


def test_unknown_institution_is_explained(df):
    r = answer(df, "how has Hogwarts University rank changed", use_gemini=False)
    assert not r.ok


def test_year_and_category_extraction(df):
    r = answer(df, "score distribution for pharmacy in 2021", use_gemini=False)
    assert r.result.params["category"] == "Pharmacy" and r.result.params["year"] == 2021
