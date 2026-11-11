"""One thin page function per analysis section; all logic lives in analysis/ via the registry."""
from __future__ import annotations

from app import ui

INTRO = {
    1: "How individual institutions and the whole field have moved since 2017: rank paths, biggest climbers and fallers, score levels and the top-10 core.",
    2: "State-level views of the top 100, how they shift over time, state-versus-state comparisons, new entrants and an India map.",
    3: "Single-variable views: how scores and ranks are distributed in a year, and how that differs between categories.",
    4: "How the five NIRF parameters relate to each other and to the overall score. Use 4.6 to plot any two variables.",
    5: "Profiles across all five parameters: who leads on what, where research outruns teaching, and who gains despite slipping on a parameter.",
    6: "Statistical comparisons: top 10 against the rest, dispersion and its change over time, which parameter varies most, and states compared.",
    7: "Movement between rank groups: promotions into the top 50, top-10 entries and exits, stability and the biggest movers.",
    8: "Statistical flags for unusual research strength, gaps between parameters, sudden score changes and volatile ranks.",
}


def _make(section: int):
    def page() -> None:
        ui.render_section(ui.load_data(), section, INTRO[section])
    page.__name__ = f"section_{section}"
    return page


PAGES = {s: _make(s) for s in range(1, 9)}
