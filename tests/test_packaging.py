"""The package metadata in pyproject.toml agrees with the code, the workflows and the docs site configuration."""

import ast
import json
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
DOCS = WORKFLOWS / "docs.yml"
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
    # The job runs the suite with the dev extra's pytest, so that floor is pinned too; ruff only lints.
    dev = [req for req in _pyproject()["project"]["optional-dependencies"]["dev"] if _name(req) != "ruff"]
    floors = _floors(_runtime_requirements() + dev)
    assert {_name(k): _version(v) for k, v in pins.items()} == {k: _version(v) for k, v in floors.items()}


def test_ci_lints_with_the_ruff_release_the_dev_extra_requires():
    # Lint results change only when ci.yml names another ruff release. The dev extra asks for at least that
    # release, since ruff 0.12 still reports 15 errors in code that 0.13 and later accept.
    if not CI.exists():
        pytest.skip("the CI workflow is not next to the tests")
    pinned = set(re.findall(r"\bruff==([0-9][0-9.]*)", CI.read_text(encoding="utf-8")))
    assert len(pinned) == 1, f"ci.yml pins ruff to {sorted(pinned) or 'no release'}"
    floor = _floors(_pyproject()["project"]["optional-dependencies"]["dev"])["ruff"]
    assert _version(floor) == _version(pinned.pop())


def test_docs_extra_caps_the_mkdocs_and_material_majors():
    # Material for MkDocs 9 builds on mkdocs 1 and gets no fixes after 5 May 2027. A major release of either
    # changes the docs toolchain, so the docs extra installs neither until the site moves on purpose.
    docs = {_name(req): REQUIREMENT.match(req).group(3) for req in _pyproject()["project"]["optional-dependencies"]["docs"]}
    assert re.search(r"<\s*2\b", docs["mkdocs"])
    assert re.search(r"<\s*10\b", docs["mkdocs-material"])


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


def test_python_classifiers_name_the_versions_ci_tests():
    # PyPI shows these classifiers, so they list exactly the Python versions the CI test matrix runs.
    if not CI.exists():
        pytest.skip("the CI workflow is not next to the tests")
    tested = set(re.findall(r'^ +- python: "(3\.[0-9]+)"', CI.read_text(encoding="utf-8"), re.M))
    classified = {c.rsplit(" :: ", 1)[1] for c in _pyproject()["project"]["classifiers"]
                  if re.fullmatch(r"Programming Language :: Python :: 3\.[0-9]+", c)}
    assert classified == tested


def test_ci_tests_each_python_release_from_the_floor_to_3_14():
    # requires-python has no upper bound, so the matrix runs every minor release from the floor to the newest
    # one it lists, 3.14 or later.
    if not CI.exists():
        pytest.skip("the CI workflow is not next to the tests")
    tested = sorted({_version(v) for v in re.findall(r'^ +- python: "(3\.[0-9]+)"', CI.read_text(encoding="utf-8"), re.M)})
    floor = _version(re.fullmatch(r">=\s*([0-9.]+)", _pyproject()["project"]["requires-python"]).group(1))
    assert tested[-1] >= (3, 14)
    assert tested == [(3, minor) for minor in range(floor[1], tested[-1][1] + 1)]


def test_ci_builds_checks_and_imports_the_distributions():
    if not CI.exists():
        pytest.skip("the CI workflow is not next to the tests")
    text = CI.read_text(encoding="utf-8")
    assert re.search(r"^  build:$", text, re.M)
    assert "twine check --strict dist/*" in text
    assert re.search(r"pip\" install dist/\*\.whl", text)


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


def _require_calls() -> set[tuple[str, str]]:
    """(module, extra) for each ``require("module", "extra")`` call in the package source."""
    calls = set()
    for path in sorted((ROOT / "src" / "viz_calc").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "require" and len(node.args) == 2
                    and all(isinstance(arg, ast.Constant) for arg in node.args)):
                calls.add((node.args[0].value, node.args[1].value))
    return calls


def test_missing_optional_dependency_names_the_package_to_install(monkeypatch):
    # The module imported (plotly.graph_objects, pptx) is not the package pip installs (plotly, python-pptx).
    from viz_calc._core import require

    calls = _require_calls()
    if not calls:
        pytest.skip("the package source is not next to the tests")
    extras = _pyproject()["project"]["optional-dependencies"]
    for module, extra in sorted(calls):
        monkeypatch.setitem(sys.modules, module, None)  # the import fails as if the package were missing
        with pytest.raises(ImportError) as info:
            require(module, extra)
        named = re.search(r"optional dependency '([^']+)'", str(info.value)).group(1)
        assert _name(named) in {_name(req) for req in extras[extra]}, f"require({module!r}, {extra!r}) names {named!r}"


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


# Runs docs/javascripts/mathjax.js in a stand-in page, once with math and once without, calls its page-load
# callback twice, and prints the src of each script element it adds.
MATHJAX_RUNNER = """
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync(process.argv[2], "utf8");
function load(hasMath) {
  const added = [];
  const callbacks = [];
  const page = {
    document: {
      querySelector: (selector) => (selector === ".arithmatex" && hasMath ? {} : null),
      getElementById: (id) => added.find((element) => element.id === id) || null,
      createElement: (tag) => ({ tag }),
      head: { appendChild: (element) => added.push(element) },
    },
    document$: { subscribe: (callback) => callbacks.push(callback) },
  };
  page.window = page;
  vm.runInNewContext(source, page);
  for (const callback of [...callbacks, ...callbacks]) callback();
  return added.map((element) => element.src);
}
console.log(JSON.stringify({ math: load(true), plain: load(false) }));
"""


def test_docs_load_mathjax_only_on_pages_with_math(tmp_path):
    # Only the methodology page has math, so no other page fetches the MathJax bundle from the CDN.
    config = ROOT / "mkdocs.yml"
    if not config.is_file():
        pytest.skip("mkdocs.yml is not next to the tests")
    scripts = re.search(r"^extra_javascript:\n((?:  - .*\n)+)", config.read_text(encoding="utf-8"), re.M).group(1)
    assert "://" not in scripts, "every page loads a script from another site"
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    (tmp_path / "runner.js").write_text(MATHJAX_RUNNER, encoding="utf-8")
    result = subprocess.run([node, str(tmp_path / "runner.js"), str(ROOT / "docs" / "javascripts" / "mathjax.js")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    added = json.loads(result.stdout)
    assert added["plain"] == []
    assert len(added["math"]) == 1 and added["math"][0].endswith("/mathjax@3/es5/tex-mml-chtml.js")


def _workflows() -> dict[str, str]:
    """{file name: text} for each workflow file."""
    if not WORKFLOWS.is_dir():
        pytest.skip("the workflows are not next to the tests")
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(WORKFLOWS.glob("*.yml"))}


def test_workflows_pin_each_action_to_a_commit():
    # A tag or a branch can be moved to other code and a commit SHA cannot. The comment names the release, so
    # an update shows in the diff.
    for name, text in _workflows().items():
        for action, ref, comment in re.findall(r"^ +(?:- )?uses: ([^@\s]+)@(\S+)(.*)$", text, re.M):
            assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{name}: {action}@{ref} is not pinned to a commit"
            assert re.fullmatch(r"  # v[0-9]+(\.[0-9]+)*", comment), f"{name}: {action} has no '  # vX.Y.Z' comment"


def test_every_workflow_job_has_a_time_limit():
    # Without timeout-minutes a job that hangs holds its runner for six hours.
    for name, text in _workflows().items():
        for job, lines in _jobs(text).items():
            assert re.search(r"^    timeout-minutes: [1-9][0-9]*$", lines, re.M), f"{name}: job {job} has no time limit"


def test_workflows_default_to_a_read_only_token():
    # A job that writes asks for it by name, so the jobs that install and run packages from PyPI can only read.
    for name, text in _workflows().items():
        assert re.search(r"^permissions:\n  contents: read\n(?!  )", text, re.M), f"{name} does not default to read"


def test_checkouts_do_not_keep_the_token():
    # The steps after a checkout install and run packages from PyPI, so the token is not left for them.
    for name, text in _workflows().items():
        for _, step in re.findall(r"^( +)- uses: actions/checkout@.*\n((?:\1  .*\n)*)", text, re.M):
            assert "persist-credentials: false" in step, f"{name}: a checkout keeps the token"


def test_jobs_that_install_the_checkout_cache_pip_downloads():
    for name, text in _workflows().items():
        for job, lines in _jobs(text).items():
            if "pip install -e" in lines:
                assert re.search(r"^ +cache: pip$", lines, re.M), f"{name}: job {job} does not cache pip downloads"


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
    assert step, f"no step is named {name!r}"
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


def _docs_workflow() -> str:
    if not DOCS.is_file():
        pytest.skip("the docs workflow is not next to the tests")
    return DOCS.read_text(encoding="utf-8")


def _condition(job: str) -> str:
    """The ``if:`` expression of a job, given as the lines below its id, with a folded block joined."""
    found = re.search(r"^    if: (?:>-\n((?:      .*\n)+)|(.*)\n)", job, re.M)
    assert found, "the job has no if: condition"
    return " ".join(line.strip() for line in found.group(1).splitlines()) if found.group(1) else found.group(2)


def _evaluate(expression: str, context: dict[str, str]) -> bool:
    """A workflow expression of ``github.*`` names, quoted strings, ``==``, ``&&``, ``||`` and parentheses."""
    python = re.sub(r"\bgithub(?:\.[\w-]+)+", lambda name: repr(context.get(name.group(0), "")), expression)
    return bool(eval(python.replace("&&", " and ").replace("||", " or "), {"__builtins__": {}}))


CI_PUSH = {"github.event_name": "workflow_run", "github.ref": "refs/heads/main",
           "github.event.workflow_run.conclusion": "success", "github.event.workflow_run.event": "push"}
BY_HAND = {"github.event_name": "workflow_dispatch", "github.ref": "refs/heads/main"}


@pytest.mark.parametrize(("event", "publishes"), [
    (CI_PUSH, True),
    ({**CI_PUSH, "github.event.workflow_run.conclusion": "failure"}, False),
    ({**CI_PUSH, "github.event.workflow_run.event": "schedule"}, False),
    ({**CI_PUSH, "github.event.workflow_run.event": "workflow_dispatch"}, False),
    ({**CI_PUSH, "github.event.workflow_run.event": "pull_request"}, False),  # from a fork's branch named main
    (BY_HAND, True),
    ({**BY_HAND, "github.ref": "refs/heads/feature"}, False),
])
def test_docs_publish_only_from_main_after_ci_passes(event, publishes):
    workflow = _docs_workflow()
    on = re.search(r"^on:\n((?:  .*\n)+)", workflow, re.M).group(1)
    assert sorted(re.findall(r"^  (\w+):", on, re.M)) == ["workflow_dispatch", "workflow_run"]
    assert "    workflows: [CI]\n    types: [completed]\n    branches: [main]\n" in on
    jobs = _jobs(workflow)
    writers = [lines for lines in jobs.values() if "contents: write" in lines]
    assert len(writers) == 1, "one job, the deploy, may write to the repository"
    assert not re.search(r"^    if:", writers[0], re.M), "the deploy runs exactly when the job it needs passed"
    gates = _needs(writers[0])
    assert len(gates) == 1
    assert _evaluate(_condition(jobs[gates[0]]), event) is publishes


def test_docs_job_that_can_push_runs_no_package_code():
    # Code from the checkout or from PyPI never runs with the token that can push to the repository.
    writers = [lines for lines in _jobs(_docs_workflow()).values() if "contents: write" in lines]
    assert len(writers) == 1
    code = "".join(line for line in writers[0].splitlines(keepends=True) if not line.lstrip().startswith("#"))
    assert not re.search(r"pip|python|mkdocs|actions/checkout|actions/setup-python", code)


def test_docs_publish_the_commit_ci_tested():
    # For a workflow_run event, github.sha is the newest commit on main, which CI may not have tested yet.
    jobs = _jobs(_docs_workflow())
    writers = [lines for lines in jobs.values() if "contents: write" in lines]
    assert len(writers) == 1
    tested = re.escape("${{ github.event.workflow_run.head_sha || github.sha }}")
    assert re.search(rf"^ +ref: {tested}( +#.*)?$", jobs[_needs(writers[0])[0]], re.M), "the build checks out another commit"
    assert re.search(rf"^ +SHA: {tested}$", writers[0], re.M), "the deploy message names another commit"


def test_docs_deploy_replaces_gh_pages_with_one_commit(tmp_path):
    # As `mkdocs gh-deploy --force --no-history` did: one commit without parents, by the owner, holding the
    # site and .nojekyll.
    workflow = _docs_workflow()
    bash, git = shutil.which("bash"), shutil.which("git")
    if bash is None or git is None:
        pytest.skip("bash or git is not installed")
    remote = tmp_path / "remote.git"
    subprocess.run([git, "init", "-q", "--bare", str(remote)], check=True)
    env = {**os.environ, "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GITHUB_REPOSITORY": "owner/repo",
           "SHA": "0123456789abcdef0123456789abcdef01234567", "TOKEN": "t0ken",
           # The push goes to the local repository instead of github.com.
           "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"url.{remote.as_uri()}.insteadOf",
           "GIT_CONFIG_VALUE_0": "https://x-access-token:t0ken@github.com/owner/repo.git"}
    script = _step_script(workflow, "Publish to gh-pages")
    for page in ("old", "new"):
        (tmp_path / page / "site").mkdir(parents=True)
        (tmp_path / page / "site" / "index.html").write_text(page, encoding="utf-8")
        result = subprocess.run([bash, "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
                                cwd=tmp_path / page, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    def remote_git(*args: str) -> str:
        return subprocess.run([git, "--git-dir", str(remote), *args], capture_output=True, text=True, check=True).stdout

    assert remote_git("log", "--format=%P|%an <%ae>|%cn|%s", "gh-pages").splitlines() == [
        "|elkronos <130187349+elkronos@users.noreply.github.com>|elkronos|Publish docs for 0123456"]
    assert remote_git("ls-tree", "--name-only", "gh-pages").split() == [".nojekyll", "index.html"]
    assert remote_git("show", "gh-pages:index.html") == "new"
