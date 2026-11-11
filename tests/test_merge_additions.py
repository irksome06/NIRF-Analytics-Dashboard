"""Tests for the pieces merged in: boundary map, Q1.6 leaderboard, institution explorer, chatbot routing for leaderboards."""
import json

import pytest

import analysis  # noqa: F401  (registers every question)
from analysis.geo import GEOJSON_PATH
from analysis.institutions import compare_institutions, institution_radar
from engine.data_loader import state_options
from engine.query_engine import run_question
from nlp.chatbot import answer


def test_geojson_covers_every_state(df):
    names = {f["properties"]["state"] for f in json.load(open(GEOJSON_PATH, encoding="utf-8"))["features"]}
    assert set(state_options(df)) <= names


@pytest.mark.parametrize("metric", ["Institutions in top 100", "All ranked institutions"])
def test_boundary_map_matches_state_table(df, metric):
    m = run_question(df, "2.6", category="Overall", year=2025, map_metric=metric, map_style="Boundary map (offline)")
    assert m.figure.data[0].type == "choropleth"
    if metric.startswith("Institutions in top"):
        t = run_question(df, "2.1", category="Overall", year=2025, top_n=30)
        assert m.data.set_index("state").institutions.to_dict() == t.data.set_index("state").institutions.to_dict()


def test_default_map_style_is_offline_boundary(df):
    assert run_question(df, "2.6").params["map_style"] == "Boundary map (offline)"


def test_other_map_styles_still_work(df):
    for style, kind in [("Basemap (needs internet)", "scattergeo"), ("Offline", "scatter")]:
        assert run_question(df, "2.6", map_style=style).figure.data[0].type == kind


def test_top_institutions_sorted_and_sized(df):
    r = run_question(df, "1.6", category="Engineering", year=2025, top_n=10)
    assert len(r.data) == 10 and r.data["position"].tolist() == [str(i) for i in range(1, 11)]
    assert r.data.iloc[0].institute_name == "Indian Institute of Technology Madras"


def test_top_institutions_unscored_category_still_lists(df):
    r = run_question(df, "1.6", category="Innovation", year=2025, top_n=5)
    assert len(r.data) == 5


@pytest.mark.parametrize("text,cat,year,n", [("top 10 engineering colleges in 2024", "Engineering", 2024, 10), ("best universities in 2023", "University", 2023, None),
                                             ("top 5 pharmacy colleges", "Pharmacy", None, 5)])
def test_chatbot_leaderboard(df, text, cat, year, n):
    r = answer(df, text, use_gemini=False)
    assert r.ok and r.result.qid == "1.6" and r.result.params["category"] == cat
    if year: assert r.result.params["year"] == year
    if n: assert r.result.params["top_n"] == n


@pytest.mark.parametrize("text,qid", [("which institutions remained consistently in the top 10", "1.5"), ("which states have the most top 100 institutions", "2.1"),
                                      ("top 10 institutions that improved the most", "1.2")])
def test_leaderboard_rule_does_not_steal_other_questions(df, text, qid):
    assert answer(df, text, use_gemini=False).result.qid == qid


def test_radar_and_compare(df):
    prof, fig = institution_radar(df, "Overall", "Indian Institute of Technology Madras", 2025)
    assert len(prof) == 5 and len(fig.data) == 3
    tbl, fig = compare_institutions(df, "Overall", ["Indian Institute of Technology Madras", "Indian Institute of Science"], 2017, 2025)
    assert len(fig.data) == 2 and set(tbl.institute_name) == {"Indian Institute of Technology Madras", "Indian Institute of Science"}
