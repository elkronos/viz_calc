# Releasing viz_calc

Publishing a GitHub release uploads the package to PyPI. The workflow
`.github/workflows/publish.yml` builds the sdist and wheel once, checks that
their description renders on PyPI, stops if the release tag is not `v`
followed by the package version, and uploads both files from the `pypi`
environment. PyPI accepts the upload through trusted publishing, so the
repository stores no PyPI token.

## One-time setup

1. Register the project name on PyPI with a pending trusted publisher. Signed
   in to pypi.org, open https://pypi.org/manage/account/publishing/, choose
   GitHub under "Add a new pending publisher" and enter:

   | Field | Value |
   |---|---|
   | PyPI project name | `viz_calc` |
   | Owner | `elkronos` |
   | Repository name | `viz_calc` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   The first upload through the pending publisher creates the project and
   makes it an ordinary trusted publisher. A pending publisher does not
   reserve the name: until that upload, anyone can create a `viz_calc`
   project, so publish the first release soon after this step.

2. Create the `pypi` environment in the repository settings: Settings →
   Environments → New environment, named `pypi`. Under "Deployment branches
   and tags", choose "Selected branches and tags" and add a tag rule `v*`, so
   only a release tag reaches the upload job. To approve each upload before it
   runs, add yourself under "Required reviewers".

## Each release

3. Prepare the release commit on `main`. In `CHANGELOG.md`, rename the
   `## Unreleased` heading to the new version, for example `## 1.2.0`. Set
   `__version__` in `src/viz_calc/__init__.py` to the same version; it is the
   only place the version is written, and `pyproject.toml` reads it from
   there. Merge the commit and wait for CI to pass on it.

   The first upload needs a version after 1.1.0, because 1.1.0 is the code
   merged in pull request #3 and `main` has changed since.

4. Tag and publish the release. On GitHub, open Releases → Draft a new
   release, type the tag `v1.2.0` (`v` and the version) under "Choose a tag"
   and create it from `main`, title the release `viz_calc 1.2.0`, paste the
   version's CHANGELOG section as the notes, and publish it.

   Publishing starts the Publish workflow on the Actions tab; approve the
   `pypi` deployment there if the environment has required reviewers. If the
   tag and `__version__` disagree, the workflow fails before anything is
   uploaded: delete the release and the tag, correct the wrong one, and
   publish again. PyPI never accepts a file name twice, even after the file is
   deleted there, so a version that reached PyPI broken is fixed with a new
   version.

## After the first upload

5. Point the documentation at PyPI and pin the README image:

   - In `README.md`, `docs/index.md` and `docs/getting-started.md`, change
     `pip install "viz_calc @ git+https://github.com/elkronos/viz_calc"` to
     `pip install viz_calc` and the `[all]` line to
     `pip install "viz_calc[all]"`. Keep one git line for installing the
     unreleased `main`.
   - Change the README image URL from `main` to the tag, for example
     `https://raw.githubusercontent.com/elkronos/viz_calc/v1.2.0/docs/images/estimation_plot.png`.
     The README is the package description on PyPI, so a pinned image stays
     as released when `docs/images` changes on `main`. In each later release
     commit, change the tag in that URL to the new one.
   - `install_hint` in `src/viz_calc/_core.py` names an extra's packages, or
     the git URL, instead of `pip install "viz_calc[extra]"`, because that
     command could fetch someone else's package while the name was free on
     PyPI. Once the project is there, its docstring and the comments in
     `tests/test_validation.py` that give this reason are out of date, and
     the git URL fallback and its tests can change to
     `pip install "viz_calc[extra]"`.

## Tags for 1.0.0 and 1.1.0

Neither earlier version was tagged. To make both installable and citable by
version from GitHub, tag the commits on `main` that match their CHANGELOG
sections and push the tags: `b2cc93d` (pull request #2, the last 1.0.0
commit) and `8fa3974` (pull request #3, which made 1.1.0).

```sh
git tag -a v1.0.0 b2cc93d -m "viz_calc 1.0.0"
git tag -a v1.1.0 8fa3974 -m "viz_calc 1.1.0"
git push origin v1.0.0 v1.1.0
```

Neither commit has the publish workflow, so a GitHub release made from either
tag uploads nothing to PyPI.
