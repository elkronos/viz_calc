"""Text the plotting functions draw: number labels, Matplotlib mathtext, Plotly markup and pickling."""

import math

import pandas as pd
import pytest

import viz_calc as vc
from viz_calc._core import abbreviate

# --- abbreviate ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "text"), [(math.inf, "inf"), (-math.inf, "-inf")])
def test_abbreviate_reads_infinity_as_inf(value, text):
    assert abbreviate(value) == text
    assert abbreviate(value, digits=0) == text


@pytest.mark.parametrize(("value", "digits", "text"), [
    (5e-324, 1, "4.9e-324"), (-5e-324, 1, "-4.9e-324"), (5e-324, 0, "5e-324"), (1.23456e-310, 2, "1.23e-310"),
    (0.034, 1, "0.034"), (0.00012345, 1, "0.00012"),
])
def test_abbreviate_gives_digits_plus_one_significant_digits_below_one(value, digits, text):
    assert abbreviate(value, digits=digits) == text


# --- Plotly markup ------------------------------------------------------------------------------------------------

LINK = '<a href="https://example.com/login">Click</a>'
FLOWS = pd.DataFrame({"from": [LINK, "A<br>B", "AT&T"], "to": ["X", "X", "<18"], "n": [3, 2, 1]})


def test_sankey_shows_node_names_as_written():
    pytest.importorskip("plotly")
    res = vc.sankey(FLOWS, source="from", target="to", value="n", title="<b>Flows</b>")
    labels = list(res.figure.data[0].node.label)
    assert labels == ['&lt;a href="https://example.com/login"&gt;Click&lt;/a&gt;', "A&lt;br&gt;B", "AT&amp;T", "X",
                      "&lt;18"]
    assert list(res.table["node"]) == [LINK, "A<br>B", "AT&T", "X", "<18"]  # the table keeps the names
    assert res.figure.layout.title.text == "<b>Flows</b>"  # the caller's own title is passed through


@pytest.mark.parametrize("directed", [False, True])
def test_interactive_network_map_shows_node_names_as_written(directed):
    pytest.importorskip("networkx")
    pytest.importorskip("plotly")
    res = vc.network_map(FLOWS, source="from", target="to", interactive=True, directed=directed)
    points = res.figure.data[-1]
    names = list(res.table["node"])
    escaped = {LINK: '&lt;a href="https://example.com/login"&gt;Click&lt;/a&gt;', "A<br>B": "A&lt;br&gt;B",
               "AT&T": "AT&amp;T", "X": "X", "<18": "&lt;18"}
    assert list(points.text) == [escaped[n] for n in names]
    for name, hover in zip(names, points.hovertext):
        assert hover.startswith(f"<b>{escaped[name]}</b><br>degree ")  # the chart's own markup stays
    assert not any("<a " in t for t in [*points.text, *points.hovertext])
