# Contributing

Thanks for considering a contribution to `pyInstanceSpace`.

## Getting set up

Use Python 3.12 and Poetry 2.3.3. Run `poetry install` from the repository root
to install the locked runtime and development dependencies. See the
[README](README.md#develop) for the main development commands.

## Before opening a PR

- Run `poetry run pytest` and make sure it passes.
- Run `poe test` (ruff, mypy `--strict`, black format checks, and pytest with coverage — all already
  configured for this project).
- Code style: ruff for linting, mypy `--strict` for type checking, black for
  formatting — no additional setup needed, just run the commands above.
- Use conventional-commit-style messages (`fix:`, `feat:`, `chore:`, ...), matching
  the existing commit history.
- If your change alters existing behaviour, add an entry to `RELEASE_NOTES.md`.

## Editing the documentation site

Curated pages live in `docs_site/src/*.md`. Keep these guides and the README
focused on current behavior. Place dated investigations and migration records
under `docs/`, outside the published website.

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
  (for example, `0.3.0` for tag `v0.3.0`).
- Publish from a GitHub Release (`published`) or use manual `workflow_dispatch` for a retry (run it on the release tag/commit).

## Reporting bugs or requesting features

Use this repository's [issue tracker](https://github.com/andremun/pyInstanceSpace/issues).

## Security issues

See `SECURITY.md` — please don't report vulnerabilities in a public issue.
