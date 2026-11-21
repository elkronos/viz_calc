"""Text the plotting functions draw: number labels, Matplotlib mathtext, Plotly markup and pickling."""

import math

import pytest

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
