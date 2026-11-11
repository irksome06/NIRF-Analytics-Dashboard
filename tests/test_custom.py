"""Chart builder: unseen questions, validation messages and a sweep over every option combination."""
import itertools

import pytest

from analysis.custom import AGGS, CHARTS, GROUPS, METRICS, SPLITS, CustomChartError, build_chart
from nlp.chatbot import answer

# (question that is not one of the 40 annexure questions, expected spec fields the chart must have)
UNSEEN = [
    ("average TLR of engineering colleges by state in 2022", dict(metric="TLR", group_by="state", category="Engineering", year_from=2022, chart="bar")),
    ("average GO of IITs over the years", dict(metric="GO", group_by="year", chart="line")),
    ("average research score of NITs by year", dict(metric="RPC", group_by="year")),
    ("which state has the highest average perception score", dict(metric="PR", group_by="state", sort="desc")),
    ("number of institutions in Kerala by year", dict(metric="count", group_by="year", states=["Kerala"])),
    ("which city has the most ranked institutions", dict(metric="count", group_by="city")),
    ("top 10 cities by number of ranked institutions in 2025", dict(metric="count", group_by="city", top_n=10, year_from=2025)),
    ("box plot of scores by category in 2024", dict(metric="score", group_by="category", chart="box")),
    ("histogram of TLR for engineering in 2024", dict(metric="TLR", chart="histogram", category="Engineering")),
    ("distribution of perception scores for universities in 2024", dict(metric="PR", chart="histogram", category="University")),
    ("lowest perception score institutions in engineering 2024", dict(metric="PR", group_by="institution", sort="asc")),
    ("top 5 universities in Maharashtra by research score", dict(metric="RPC", group_by="institution", top_n=5, states=["Maharashtra"])),
    ("compare teaching and perception for pharmacy in 2023", dict(metric="TLR", metric_y="PR", chart="scatter", category="Pharmacy")),
    ("compare average outreach of engineering and management colleges over time", dict(metric="OI", group_by="year", split_by="category")),
    ("heatmap of average score by state and category for engineering and university 2024", dict(metric="score", group_by="state", chart="heatmap")),
    ("how many IITs are in the top 50 each year", dict(metric="count", group_by="year", rank_max=50)),
    ("number of colleges in each state in 2023", dict(metric="count", group_by="state", category="College")),
    ("show outreach by state for management colleges", dict(metric="OI", group_by="state", category="Management")),
    ("which category has the highest average research score in 2024", dict(metric="RPC", group_by="category")),
]


@pytest.mark.parametrize("text,expect", UNSEEN)
def test_unseen_question_builds_a_matching_chart(df, text, expect):
    r = answer(df, text, use_gemini=False)
    assert r.ok, r.message
    assert r.result.qid == "custom", (text, r.result.title)
    assert len(r.result.data) > 0 and len(r.result.figure.data) > 0
    for k, v in expect.items():
        assert r.result.params.get(k) == v, (text, k, r.result.params)


def test_follow_ups_reuse_the_previous_custom_chart(df):
    first = answer(df, "average TLR of engineering colleges by state in 2022", use_gemini=False)
    ctx = {"last_query": first.resolved}
    r = answer(df, "what about pharmacy?", context=ctx, use_gemini=False)
    assert r.ok and r.result.params["category"] == "Pharmacy" and r.result.params["group_by"] == "state" and r.result.params["metric"] == "TLR"
    r = answer(df, "now 2024", context=ctx, use_gemini=False)
    assert r.ok and r.result.params["year_from"] == 2024 and r.result.params["category"] == "Engineering"
    r = answer(df, "show it by year", context=ctx, use_gemini=False)
    assert r.ok and r.result.params["group_by"] == "year" and r.result.params["metric"] == "TLR"


def test_registered_questions_are_not_taken_over(df):
    for text, qid in [("Which states have the highest number of institutions in the top 100?", "2.1"), ("How has IIT Madras ranking changed from 2017 to 2025?", "1.1"),
                      ("Which institutions have unusually high research scores relative to overall?", "8.1")]:
        r = answer(df, text, use_gemini=False)
        assert r.ok and r.result.qid == qid, (text, r.result.qid)


@pytest.mark.parametrize("text", ["what is the capital of india", "who will win the world cup", "tell me a joke"])
def test_off_topic_is_declined(df, text):
    assert not answer(df, text, use_gemini=False).ok


@pytest.mark.parametrize("spec,fragment", [
    (dict(metric="foo"), "Metric must be"),
    (dict(metric="score", category="Dental", year=2017), "editions 2020-2025"),
    (dict(metric="score", year=1999), "outside that range"),
    (dict(metric="score", states=["Atlantis"]), "State must be"),
    (dict(metric="count", metric_y="TLR"), "scatter"),
    (dict(metric="score", chart="scatter"), "second measure"),
    (dict(metric="score", chart="box"), "grouping"),
    (dict(metric="score", surprise=1), "Unsupported"),
])
def test_bad_requests_get_a_clear_message(df, spec, fragment):
    with pytest.raises(CustomChartError) as e:
        build_chart(df, spec)
    assert fragment in str(e.value)


def test_every_option_combination_either_draws_or_explains(df):
    """No combination of measure, grouping, split and chart may raise anything except a friendly CustomChartError."""
    drawn = 0
    for metric, group, split, chart in itertools.product(list(METRICS), [None, *GROUPS], [None, *SPLITS], CHARTS):
        spec = dict(metric=metric, group_by=group, split_by=split, chart=chart, category=None if "category" in (group, split) else "Overall",
                    categories=["Engineering", "University"] if "category" in (group, split) else None, year_from=2022, year_to=2025,
                    metric_y="RPC" if chart == "scatter" else None)
        try:
            data, fig, notes, clean = build_chart(df, spec)
        except CustomChartError:
            continue
        assert len(data) > 0 and len(fig.data) > 0
        drawn += 1
    assert drawn > 150


@pytest.mark.parametrize("agg", list(AGGS))
def test_every_aggregation(df, agg):
    data, fig, notes, clean = build_chart(df, dict(metric="TLR", agg=agg, group_by="state", category="Engineering", year=2024))
    assert len(data) > 0


def test_name_family_and_rank_cut_filters(df):
    data, _, _, clean = build_chart(df, dict(metric="count", group_by="year", name_contains="IIT", rank_max=20))
    assert clean["name_contains"] == "Indian Institute of Technology" and data["Institutions"].max() <= 20
