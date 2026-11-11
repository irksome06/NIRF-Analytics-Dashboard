"""Regression tests for bugs found in code review."""
from nlp.chatbot import answer


def test_rules_path_reaches_the_chart_builder(df):
    """parse_custom must be called on the rules path, not only when Gemini is on."""
    r = answer(df, "average TLR of engineering colleges by state in 2022", use_gemini=False)
    assert r.ok and r.result.qid == "custom"


def test_registered_followup_is_not_read_as_custom_chart(df):
    r1 = answer(df, "top 10 engineering colleges in 2024", use_gemini=False)
    r2 = answer(df, "what about pharmacy?", context={"last_query": r1.resolved}, use_gemini=False)
    assert r2.ok and r2.result.params["category"] == "Pharmacy"


def test_state_share_uses_the_national_total_not_the_rows_shown(df):
    shares = set()
    for n in (5, 10, 20):
        r = answer(df, f"Which states have the most institutions in the top 100 in 2025? top {n} states", use_gemini=False)
        shares.add(r.result.data["share_pct"].iloc[0])
    assert shares == {17.0}


# ---------------------------------------------------------------- full / readable institution names
def test_short_name_never_cuts_mid_word_for_any_institution(df):
    from engine.helpers import short_name
    for n in df["institute_name"].unique():
        base = short_name(n, None)
        out = short_name(n, 28)
        assert len(out) <= 28
        if out.endswith("…"):
            cut = out[:-1]
            assert base.startswith(cut) and (len(base) == len(cut) or base[len(cut)] in " ,-("), (n, out)


def test_known_families_are_abbreviated_readably():
    from engine.helpers import short_name
    assert short_name("Indian Institute of Technology Madras", 28) == "IIT Madras"
    assert short_name("Indian Institute of Science", 28) == "IISc"
    assert short_name("Birla Institute of Technology & Science -Pilani", 28) == "BITS Pilani"
    assert short_name("Indian Institute of Technology (Banaras Hindu University) Varanasi", 28) == "IIT (BHU) Varanasi"
    assert short_name("Jawaharlal Nehru University", 40) == "Jawaharlal Nehru University"


def test_kpi_cards_show_full_institution_names(df):
    names = set(df["institute_name"])
    for q in ("top 10 engineering colleges in 2024", "Which engineering institutions improved the most since 2018?",
              "Which institutions declined the most between 2018 and 2025?", "who is ranked first in overall 2025"):
        r = answer(df, q, use_gemini=False)
        for m in (r.metrics or []):
            v = str(m["value"])
            assert not any(n != v and n.startswith(v) for n in names if len(v) >= 12), (q, v)


def test_scatter_labels_are_short_but_hover_has_full_names(df):
    from engine.query_engine import run_query
    for qid, extra in (("5.2", {}), ("8.1", {}), ("8.5", {})):
        fig = run_query(df, {"question": qid, "category": "Overall", **extra}).figure
        for tr in fig.data:
            if tr.mode == "markers+text" and tr.customdata is not None:
                for lab, full in zip(tr.text, tr.customdata):
                    assert lab and len(lab) <= 28 and (lab == full or len(full) > len(lab) - 1)
                    assert not lab.endswith("Techno")
                assert "%{customdata}" in tr.hovertemplate


def test_scatter_highlights_show_full_names():
    """Highlighted scatter points are labelled with the complete institution name (no clipping, no abbreviation)."""
    from engine.query_engine import run_question
    from engine.data_loader import load_master
    df = load_master()
    for qid in ("5.2", "8.1", "8.4", "8.5"):
        res = run_question(df, qid)
        notes = [a.text.replace("<br>", " ") for a in res.figure.layout.annotations if a.showarrow]
        names = list(res.data["institute_name"])
        assert len(notes) == len(names)
        assert sorted(notes) == sorted(" ".join(n.split()) for n in names)
