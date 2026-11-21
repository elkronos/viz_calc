"""The package metadata in pyproject.toml agrees with the code and with the CI workflow."""

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; pytest depends on tomli there
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
RUNTIME_EXTRAS = ("network", "interactive", "pptx")  # "all" combines these; "dev" and "docs" are tooling
REQUIREMENT = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)")


def _pyproject() -> dict:
    path = ROOT / "pyproject.toml"
    if not path.exists():
        pytest.skip("pyproject.toml is not next to the tests")
    with path.open("rb") as f:
        return tomllib.load(f)


def _name(requirement: str) -> str:
    """The normalised distribution name of a requirement string (PEP 503)."""
    return re.sub(r"[-_.]+", "-", REQUIREMENT.match(requirement).group(1)).lower()


def _version(text: str) -> tuple[int, ...]:
    """A release number as a tuple without trailing zeros, so 2.0 and 2.0.0 compare equal."""
    parts = [int(part) for part in text.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _floors(requirements: list[str]) -> dict[str, str]:
    """{name: minimum version} for each requirement with a ``>=`` bound."""
    floors = {}
    for req in requirements:
        if _name(req) == "viz-calc":
            continue
        bound = re.search(r">=\s*([0-9][0-9.]*)", REQUIREMENT.match(req).group(3))
        if bound:
            floors[_name(req)] = bound.group(1)
    return floors


def _runtime_requirements() -> list[str]:
    project = _pyproject()["project"]
    extras = project["optional-dependencies"]
    return list(project["dependencies"]) + [req for extra in RUNTIME_EXTRAS for req in extras[extra]]


def _ci_entry(label: str) -> dict[str, str]:
    """The keys of the CI test matrix entry with this label, read from ci.yml."""
    if not CI.exists():
        pytest.skip("the CI workflow is not next to the tests")
    # An entry is a "- python:" line and the lines below it that are indented further.
    for _, entry in re.findall(r"^( +)- (python:.*\n(?:\1  .*\n)*)", CI.read_text(encoding="utf-8"), re.M):
        keys = dict(re.findall(r'^\s*([a-z]+): "?([^"\n]*?)"?$', entry, re.M))
        if keys.get("label") == label:
            return keys
    raise AssertionError(f"ci.yml has no test matrix entry labelled {label!r}")


def test_ci_minimum_job_installs_exactly_the_declared_floors():
    pins = dict(p.split("==") for p in _ci_entry("minimum versions")["deps"].split())
    floors = _floors(_runtime_requirements())
    assert {_name(k): _version(v) for k, v in pins.items()} == {k: _version(v) for k, v in floors.items()}


def test_numpy_floor_skips_1_24_0():
    # NumPy 1.24.0's np.ma.masked_invalid raises TypeError for a pandas Series, which Matplotlib's
    # fill_between passes it, so timeseries_fill failed with a numeric time column. 1.24.1 fixed it.
    floor = _floors(_pyproject()["project"]["dependencies"])["numpy"]
    assert _version(floor) >= (1, 24, 1)


def test_build_backend_floor_writes_the_license_fields():
    # hatchling before 1.26 rejects the list form of project.license-files, and 1.26 writes the legacy
    # License field. 1.27 is the first release that writes License-Expression and License-File.
    data = _pyproject()
    assert isinstance(data["project"]["license-files"], list)
    assert _version(_floors(data["build-system"]["requires"])["hatchling"]) >= (1, 27)
