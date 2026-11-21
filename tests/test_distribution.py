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
