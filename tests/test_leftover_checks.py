"""Role, infinite-value, constant-metric and missing-group checks in the time series and multivariate charts."""

import matplotlib.pyplot as plt
import pytest

import viz_calc as vc
from viz_calc import datasets


@pytest.fixture(scope="module")
def meas():
    return datasets.measurements()


# --- one column passed for two roles --------------------------------------------------------------------------------


def test_radar_rejects_a_group_that_is_also_a_metric(meas):
    d = meas.assign(size=meas["length"].round())
    with pytest.raises(ValueError, match="group and metrics must be different columns; both are 'size'"):
        vc.radar(d, metrics=["length", "width", "size"], group="size")
    assert plt.get_fignums() == []
