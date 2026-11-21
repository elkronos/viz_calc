"""The package metadata in pyproject.toml agrees with the code and with the CI and publish workflows."""

import ast
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; pytest depends on tomli there
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
PUBLISH = WORKFLOWS / "publish.yml"
TAG_CHECK = "Check the release tag against the package version"  # a step of publish.yml
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


def _publish_workflow() -> str:
    if not WORKFLOWS.is_dir():
        pytest.skip("the workflows are not next to the tests")
    assert PUBLISH.is_file(), "no workflow uploads a release to PyPI"
    return PUBLISH.read_text(encoding="utf-8")


def _jobs(workflow: str) -> dict[str, str]:
    """{job id: the lines below it} for each job of a workflow file."""
    return dict(re.findall(r"^  ([\w-]+):\n((?:    .*\n|\n)*)", workflow.split("\njobs:\n", 1)[1], re.M))


def _needs(job: str) -> list[str]:
    """The ids of the jobs that a job, given as the lines below its id, waits for."""
    needs = re.search(r"^    needs: \[?([\w, -]*?)\]?\n", job, re.M)
    return [name.strip() for name in needs.group(1).split(",")] if needs else []


def _step_script(workflow: str, name: str) -> str:
    """The ``run`` script of the workflow step with this name."""
    step = re.search(rf"^( +)- name: {re.escape(name)}\n((?:\1  .*\n)*)", workflow, re.M)
    assert step, f"{PUBLISH.name} has no step named {name!r}"
    run = re.search(r"^( +)run: \|\n((?:\1  .*\n|\n)*)", step.group(2), re.M)
    assert run, f"the step {name!r} has no run script"
    return textwrap.dedent(run.group(2))


def test_publish_workflow_uploads_a_release_from_the_pypi_environment_only():
    # PyPI's trusted publisher names this workflow file and environment, and RELEASING.md tells the owner both.
    workflow = _publish_workflow()
    assert re.search(r"^on:\n  release:\n    types: \[published\]\n", workflow, re.M)
    assert re.search(r"^permissions:\n  contents: read\n", workflow, re.M)
    jobs = _jobs(workflow)
    signing = [job for job, lines in jobs.items() if "id-token: write" in lines]
    assert len(signing) == 1, "one job, the upload, may request the token PyPI accepts"
    upload = jobs[signing[0]]
    assert "pypa/gh-action-pypi-publish@" in upload
    assert "actions/checkout" not in upload, "the upload job uploads what the build job made"
    environment = re.search(r"^    environment:\n      name: (\S+)\n", upload, re.M)
    build = [jobs[job] for job in _needs(upload) if "python -m build" in jobs[job]]
    assert environment and build
    assert "twine check --strict dist/*" in build[0] and TAG_CHECK in build[0]
    releasing = (ROOT / "RELEASING.md").read_text(encoding="utf-8")
    assert f"`{PUBLISH.name}`" in releasing and f"`{environment.group(1)}`" in releasing


def test_publish_workflow_uploads_only_after_the_tests_pass_on_the_built_wheel():
    # The upload waits for the test suite to pass with the uploaded wheel and its dependencies installed.
    jobs = _jobs(_publish_workflow())
    upload = next(lines for lines in jobs.values() if "pypa/gh-action-pypi-publish@" in lines)
    tests = [jobs[job] for job in _needs(upload) if re.search(r"^ +run: pytest\n", jobs[job], re.M)]
    assert tests, "the upload job does not wait for a job that runs the tests"
    assert "actions/download-artifact" in tests[0] and re.search(r"pip install .*dist/\*\.whl", tests[0])
    assert "pip install -e" not in tests[0]


@pytest.mark.parametrize(("tag", "passes"), [("v2.3.4", True), ("v2.3.5", False), ("2.3.4", False)])
def test_publish_workflow_stops_when_the_tag_is_not_v_and_the_version(tmp_path, tag, passes):
    workflow = _publish_workflow()
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "viz_calc-2.3.4-py3-none-any.whl").touch()
    (tmp_path / "dist" / "viz_calc-2.3.4.tar.gz").touch()
    # GitHub runs a step's script with these bash options.
    result = subprocess.run([bash, "--noprofile", "--norc", "-eo", "pipefail", "-c", _step_script(workflow, TAG_CHECK)],
                            cwd=tmp_path, env={**os.environ, "TAG": tag}, capture_output=True, text=True)
    assert (result.returncode == 0) is passes, result.stdout + result.stderr
    if not passes:
        assert "v2.3.4" in result.stdout
