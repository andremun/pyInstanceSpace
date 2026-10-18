# Contributing

Contributor workflow for `pyInstanceSpace`.

## Getting set up

Use Python 3.12 and Poetry 2.3.3. Run `poetry install` from the repository root
to install the locked runtime and development dependencies. See the
[README](README.md#develop) for the main development commands.

## Before opening a PR

- Classify the change as **Additive** or **Behavior-changing**. Explain affected
  callers and scientific before/after validation for behavior changes; include
  the compatibility impact in `RELEASE_NOTES.md`.
- Run checks appropriate to the change. `poetry run poe test` runs Ruff, strict
  mypy, Black, the checked documentation build and pytest with coverage. The
  [test guide](tests/README.md) gives the strict release-gate commands.
- Use conventional commit messages (`fix:`, `feat:`, `docs:`, `chore:`).

## Editing the documentation site

Curated pages live in `docs_site/src/*.md`. Keep these guides and the README
focused on current behavior. Place dated investigations and migration records
under `docs/` only when they remain necessary reproducibility evidence. Use
GitHub Issues for active tasks and Git for completed development history.

`docs_site/options.json` contains option descriptions. Update the corresponding
source dataclass docstrings as well. Names, defaults and JSON aliases are read
from the live schema; the build fails for missing or stale option descriptions.
`docs_site/templates/`, `docs_site/pdoc-template/` and `docs_site/assets/` provide
the shared layout for curated pages and generated API pages. Preserve existing
API paths and member anchors.

Run `poetry run poe docs` to build and check the complete site in `site/`, then
serve it with `python -m http.server --directory site`. Check desktop and narrow
mobile layouts, keyboard search and navigation, code examples and option tables.
`tests/test_documentation.py` executes published examples and checks schema
coverage, aliases, links and retained API anchors. The Pages workflow builds a
preview artifact for pull requests and deploys only from `main`.

## Before publishing a release

- Confirm `[tool.poetry].version` in `pyproject.toml` matches the release tag
  (`0.4.1` corresponds to tag `v0.4.1`).
- Publish from a GitHub Release (`published`) or use manual `workflow_dispatch` for a retry (run it on the release tag/commit).

Current contributor references are indexed in [docs/README.md](docs/README.md).
Follow [the release procedure](docs/release-0.4.1.md) for the release under review.
Record compatibility impacts and validation in the PR rather than adding a
completed-task log or a duplicate local backlog.

## Reporting bugs or requesting features

Use this repository's [issue tracker](https://github.com/andremun/pyInstanceSpace/issues).

## Security issues

See `SECURITY.md` — please don't report vulnerabilities in a public issue.
