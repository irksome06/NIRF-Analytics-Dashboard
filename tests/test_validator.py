import pytest

from engine.query_engine import AmbiguousInstitution, QueryValidationError, run_query, validate_query


@pytest.mark.parametrize("bad", [
    {"question": "99.9"}, {"intent": "hack"}, {"question": "3.1", "category": "Astrology"},
    {"question": "3.1", "year": 1999}, {"question": "3.1", "drop_table": 1}, "not a dict",
    {"question": "4.6", "x": "salary"}, {"question": "2.4", "state_a": "Atlantis"}, {"question": "1.1"},
])
def test_rejects_invalid(df, bad):
    with pytest.raises(QueryValidationError):
        validate_query(df, bad)


def test_ambiguous_institution_lists_candidates(df):
    with pytest.raises(AmbiguousInstitution):
        validate_query(df, {"question": "1.1", "category": "Engineering", "institution": "iit"})


def test_intent_only_uses_default_question(df):
    assert validate_query(df, {"intent": "map"})[0] == "2.6"


def test_year_order_is_repaired(df):
    _, p = validate_query(df, {"question": "1.2", "year_from": 2025, "year_to": 2018})
    assert p["year_from"] == 2018 and p["year_to"] == 2025


def test_numeric_values_are_clamped(df):
    _, p = validate_query(df, {"question": "8.3", "z": 99})
    assert p["z"] == 4.0
