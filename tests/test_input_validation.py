"""Input checks shared by the charts: sums of nullable numbers."""

import numpy as np
import pandas as pd
import pytest

import viz_calc as vc
from viz_calc._core import safe_sum

# --- sums of nullable numbers come back in NumPy dtypes -------------------------------------------------------------


@pytest.mark.parametrize("values, dtype, expected", [
    ([1, 2, None, 4], "Int64", np.int64),
    ([1, 2, 3, 4], "Int64", np.int64),
    ([1.5, 2.0, None, 4.0], "Float64", np.float64),
], ids=["Int64 with NA", "Int64 complete", "Float64 with NA"])
def test_waffle_and_nested_pie_tables_hold_numpy_numbers(values, dtype, expected):
    d = pd.DataFrame({"c": list("abab"), "o": list("xxyy"), "v": pd.array(values, dtype=dtype)})
    waffle = vc.waffle(d, category="c", value="v", order=["a", "b"]).table
    nested = vc.nested_pie(d, outer="o", inner="c", value="v").table
    for table in (waffle, nested):
        assert table["value"].dtype == expected
        assert "value" in table.select_dtypes("number").columns
    total = np.nansum(np.array(values, dtype=float))
    assert waffle["value"].sum() == nested["value"].sum() == total


def test_safe_sum_groups_by_several_keys_and_keeps_missing_keys():
    v = pd.Series([1, 2, None, 4], dtype="Int64")
    a, b = pd.Series(["x", "x", None, "y"]), pd.Series(["p", "q", "p", None])
    kept = safe_sum(v, [a, b], dropna=False)
    assert kept.dtype == np.int64 and kept.sum() == 7 and len(kept) == 4
    dropped = safe_sum(v, [a, b])
    assert dropped.dtype == np.int64 and dropped.to_dict() == {("x", "p"): 1, ("x", "q"): 2}
