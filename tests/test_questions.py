import pytest

from engine.query_engine import QUESTIONS, default_params, run_query

IDS = sorted(QUESTIONS, key=lambda s: [int(x) for x in s.split(".")])
CATS = ["Overall", "Engineering", "University", "College", "Pharmacy", "Dental"]


def test_all_forty_annexure_questions_registered():
    expected = {f"{s}.{i}" for s, n in zip(range(1, 9), [5] * 8) for i in range(1, n + 1)}
    assert expected <= set(QUESTIONS)


@pytest.mark.parametrize("cat", CATS)
@pytest.mark.parametrize("qid", IDS)
def test_question_runs_and_returns_contract(df, qid, cat):
    p = default_params(df, qid)
    p["category"] = cat
    if "institution" in p:
        p["institution"] = df[(df["category"] == cat) & (df["rank_status"] == "exact")]["institution_key"].iloc[0]
    years = sorted(df[df["category"] == cat]["year"].unique())
    for k in ("year", "year_to", "year_b"):
        if k in p:
            p[k] = int(years[-1])
    r = run_query(df, {"question": qid, **p})
    assert hasattr(r.data, "columns") and r.figure is not None


def test_top100_state_counts_match_list_size(df):
    r = run_query(df, {"question": "2.1", "category": "Engineering", "year": 2025, "top_n": 30})
    assert 0 < r.data["institutions"].sum() <= 100


def test_band_to_top50_is_subset_of_51_100(df):
    r = run_query(df, {"question": "7.1", "category": "Overall", "year_a": 2024, "year_b": 2025})
    assert (r.data.iloc[:, 1] == "51–100").all() and r.data.iloc[:, 2].isin(["Top 10", "11–50"]).all()


def test_score_trend_keeps_weight_schemes_apart(df):
    r = run_query(df, {"question": "1.4", "category": "College"})
    assert r.data["score_comparison_group"].nunique() >= 2
