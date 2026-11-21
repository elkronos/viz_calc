"""Calls that warned on part of the supported pandas, NumPy and Matplotlib range draw without warnings.

pyproject.toml turns FutureWarning and viz_calc's own DeprecationWarning into errors. These tests also
fail on a DeprecationWarning that Matplotlib or NumPy raise for an argument viz_calc passes.
"""

import warnings

import pandas as pd
import pytest

import viz_calc as vc


def test_centered_bar_with_one_group_warns_nothing():
    # pandas 2.1 to 2.3 warned when Matplotlib called float() on a one-element Series of error bar sizes.
    d = pd.DataFrame({"g": ["a"] * 4, "v": [1.0, 2, 3, 4]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = vc.centered_bar(d, "g", "v")
    assert res.table[["n", "n_above"]].values.tolist() == [[4, 2]]


def test_sankey_with_categorical_endpoints_warns_nothing():
    # pandas 2.1 to 2.3 warned that the default of groupby(observed=False) on a Categorical will change.
    pytest.importorskip("plotly")
    flows = pd.DataFrame({"a": pd.Categorical(["x", "y"], categories=["x", "y", "u"]),
                          "b": pd.Categorical(["y", "z"], categories=["y", "z", "v"]), "f": [1, 2]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = vc.sankey(flows, "a", "b", "f")
    assert res.table.to_dict("list") == {"node": ["x", "y", "z"], "inflow": [0, 1, 2], "outflow": [1, 2, 0]}
