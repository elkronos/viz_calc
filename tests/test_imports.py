"""What importing the package loads: scipy.stats waits for the first statistic."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import viz_calc

PACKAGE = Path(viz_calc.__file__).resolve().parent


def _run(code: str) -> list[str]:
    """Output lines of *code* run in a fresh interpreter that imports this copy of viz_calc."""
    env = dict(os.environ, MPLBACKEND="Agg")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PACKAGE.parent), env.get("PYTHONPATH")]))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          check=True).stdout.splitlines()


@pytest.mark.parametrize("module", ["viz_calc", "viz_calc.datasets", "viz_calc.stats"])
def test_import_does_not_load_scipy_stats(module):
    # scipy.stats takes about as long to import as pandas and Matplotlib together.
    out = _run(f"import pathlib, sys, {module}; print(pathlib.Path(sys.modules['viz_calc'].__file__).resolve()); "
               "print('scipy.stats' in sys.modules)")
    assert out == [str(PACKAGE / "__init__.py"), "False"]


def test_first_statistic_loads_scipy_stats():
    out = _run("import sys, viz_calc as vc; print('scipy.stats' in sys.modules); "
               "print(vc.stats.bootstrap_ci([1.0, 2.0, 4.0, 7.0], n_resamples=200, seed=0)[0]); "
               "print('scipy.stats' in sys.modules)")
    assert out == ["False", "3.5", "True"]
