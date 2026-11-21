"""Histogram bin rules and range-checked options of the distribution charts."""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets

# --- options checked with check_range are used as the float it returns --------------------------------------------


@pytest.mark.parametrize("value", [Decimal("0.5"), Fraction(1, 2), np.float32(0.5), np.array(0.5)])
def test_histogram_alpha_accepts_any_real_number(value):
    meas = datasets.measurements()
    bars = vc.histogram(meas, "length", alpha=value).axes[0, 0].patches
    assert bars and all(b.get_alpha() == 0.5 for b in bars)


@pytest.mark.parametrize("value", [Decimal("0.5"), Fraction(1, 2), np.float32(0.5), np.array(0.5)])
def test_ridgeplot_overlap_accepts_any_real_number(value):
    meas = datasets.measurements()
    expected = vc.ridgeplot(meas, "length", "group", overlap=0.5).axes.get_yticks()
    assert list(vc.ridgeplot(meas, "length", "group", overlap=value).axes.get_yticks()) == list(expected)


@pytest.mark.parametrize("value", [Decimal(100), Fraction(100), np.float32(100), np.array(100)])
def test_to_pptx_dpi_accepts_any_real_number(value, tmp_path):
    pytest.importorskip("pptx")
    res = vc.histogram(datasets.measurements(), "length")
    assert vc.to_pptx(res, str(tmp_path / "a.pptx"), dpi=value) == str(tmp_path / "a.pptx")


# --- the bin rule reported for a column with a single value ------------------------------------------------------


@pytest.mark.parametrize("rule", ["fd", "auto"])
@pytest.mark.parametrize("value", [3, 3.0, 0.1])
def test_a_constant_column_reports_the_sturges_fallback(rule, value):
    d = pd.DataFrame({"v": [value] * 6})
    res = vc.histogram(d, "v", bins=rule)
    assert res.info["bin_rule"] == "sturges (IQR is 0, so FD is undefined)"
    assert res.info["bin_edges"].tolist() == [value - 0.5, value + 0.5]
    assert res.table["n"].tolist() == [6]
    near = vc.histogram(pd.DataFrame({"v": [value] * 6 + [value + 1]}), "v", bins=rule)  # IQR 0, two values
    assert near.info["bin_rule"] == res.info["bin_rule"]
