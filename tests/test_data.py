from engine.data_loader import PARAMS, filter_data, resolve_institution, validate


def test_master_is_valid(df):
    assert validate(df) == []
    assert df["year"].min() == 2017 and df["year"].max() == 2025


def test_no_duplicate_rows(df):
    assert not df.duplicated(["institution_key", "year", "category"]).any()


def test_scores_in_range(df):
    for c in ["score", *PARAMS]:
        assert df[c].dropna().between(0, 100).all()


def test_filter_year_range_and_exact(df):
    d = filter_data(df, years=(2020, 2022), category="Engineering", exact_only=True)
    assert d["year"].between(2020, 2022).all() and (d["rank_status"] == "exact").all()


def test_resolver_aliases(df):
    assert resolve_institution(df, "iit madras")["institute_name"].tolist() == ["Indian Institute of Technology Madras"]
    assert "Hyderabad" in resolve_institution(df, "iiit hyderabad")["institute_name"].iloc[0]
    assert len(resolve_institution(df, "bits pilani")) == 1
    assert len(resolve_institution(df, "iit")) > 10          # bare acronym is ambiguous, not wrong
    assert resolve_institution(df, "zzzz nonsense").empty
