"""The package metadata in pyproject.toml agrees with the code and with the CI workflow."""

import ast
import re
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; pytest depends on tomli there
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
RUNTIME_EXTRAS = ("network", "interactive", "pptx")  # "all" combines these; "dev" and "docs" are tooling
DISTRIBUTIONS = {"dateutil": "python-dateutil", "PIL": "pillow", "pptx": "python-pptx"}  # import name -> package
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


def test_version_is_written_once():
    # The build reads the version from __version__, so a release bump cannot change one copy and miss the other.
    import viz_calc

    data = _pyproject()
    assert "version" not in data["project"], "pyproject.toml repeats the version written in __version__"
    assert "version" in data["project"].get("dynamic", [])
    source = ROOT / data["tool"]["hatch"]["version"]["path"]
    assert source == ROOT / "src" / "viz_calc" / "__init__.py"
    # hatchling's default pattern for a version file: the build takes the first match and fails without one.
    pattern = re.compile(r"""^(?:__version__|VERSION) *= *(['"])v?(.+?)\1""", re.M | re.I)
    found = [version for _, version in pattern.findall(source.read_text(encoding="utf-8"))]
    assert found == [viz_calc.__version__]


def test_build_backend_floor_writes_the_license_fields():
    # hatchling before 1.26 rejects the list form of project.license-files, and 1.26 writes the legacy
    # License field. 1.27 is the first release that writes License-Expression and License-File.
    data = _pyproject()
    assert isinstance(data["project"]["license-files"], list)
    assert _version(_floors(data["build-system"]["requires"])["hatchling"]) >= (1, 27)


def _third_party_imports() -> dict[str, str]:
    """{top-level module: a place it is imported} for each non-stdlib module the package imports.

    A module loaded with ``require("module", extra)`` counts as imported.
    """
    found = {}
    for path in sorted((ROOT / "src" / "viz_calc").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            elif (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "require"
                  and node.args and isinstance(node.args[0], ast.Constant)):
                names = [node.args[0].value]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top != "viz_calc":
                    found.setdefault(top, f"{path.name}:{node.lineno}")
    return found


def test_every_third_party_import_is_a_declared_dependency():
    imports = _third_party_imports()
    if not imports:
        pytest.skip("the package source is not next to the tests")
    declared = {_name(req) for req in _runtime_requirements()}
    undeclared = {where: module for module, where in imports.items()
                  if _name(DISTRIBUTIONS.get(module, module)) not in declared}
    assert not undeclared, f"imported but not in pyproject.toml dependencies or extras: {undeclared}"


def test_readme_images_load_from_the_package_page():
    # README.md is the package's long description, and a package index resolves a relative src against
    # its own site, so each image is the raw.githubusercontent.com URL of a file in this repository.
    project = _pyproject()["project"]
    text = (ROOT / project["readme"]).read_text(encoding="utf-8")
    sources = re.findall(r'<img[^>]*\ssrc="([^"]+)"', text) + re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text)
    assert sources, "README.md has no images"
    url = re.compile(r"https://raw\.githubusercontent\.com/elkronos/viz_calc/[^/]+/(.+)")  # main or a release tag
    for src in sources:
        match = url.fullmatch(src)
        assert match, f"README.md image {src!r} is not a raw.githubusercontent.com URL of this repository"
        assert (ROOT / match.group(1)).is_file(), f"README.md image {src!r} is not in the repository"
