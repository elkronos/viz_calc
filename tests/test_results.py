"""The result object, saving and export, and how a call treats the caller's Axes and figures."""

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets

SMALL = pd.DataFrame({"k": list("abcd"), "a": [3.0, 5, 2, 8], "b": [4.0, 1, 6, 7]})
METRICS = ["length", "width", "depth", "mass"]


# --- an Axes in a SubFigure -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("draw, projection", [
    (lambda ax: vc.lollipop(datasets.survey(), x="team", y="tenure_years", ax=ax), None),
    (lambda ax: vc.correlogram(datasets.measurements(), ax=ax), None),
    (lambda ax: vc.bullet(SMALL.head(1), label="k", value="a", target="b", ax=ax), None),
    (lambda ax: vc.radar(datasets.measurements(), metrics=METRICS, group="group", ax=ax), "polar"),
])
def test_result_of_an_axes_in_a_subfigure_holds_the_whole_figure(draw, projection, tmp_path):
    fig = plt.figure(figsize=(10, 4))
    left, right = fig.subfigures(1, 2)
    title = right.suptitle(r"$\alpha$ panel")
    res = draw(left.add_subplot(projection=projection))
    assert res.figure is fig
    buf = BytesIO()
    assert res.save(buf, format="png") is buf and buf.getvalue().startswith(b"\x89PNG")
    assert res.save(str(tmp_path / "chart")).endswith("chart.png")
    assert title.get_parse_math()  # text in the other panel is left alone


def test_to_pptx_takes_a_result_or_a_subfigure(tmp_path):
    pptx = pytest.importorskip("pptx")
    fig = plt.figure()
    left, _ = fig.subfigures(1, 2)
    res = vc.lollipop(datasets.survey(), x="team", y="tenure_years", ax=left.subplots())
    out = vc.to_pptx([res, left], str(tmp_path / "deck.pptx"))
    assert len(pptx.Presentation(out).slides) == 2


def test_a_result_made_from_a_subfigure_keeps_the_whole_figure():
    fig = plt.figure()
    sub = fig.subfigures(1, 2)[0].subfigures(2, 1)[1]  # nested
    assert vc.VizResult(sub).figure is fig
    assert vc.VizResult(fig).figure is fig
