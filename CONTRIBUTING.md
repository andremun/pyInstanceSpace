# Contributing

Thanks for considering a contribution to `pyInstanceSpace`.

## Getting set up

Follow the *Development Environment Setup Guide* in `README.md` to install Poetry and
the project's dependencies.

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

`docs_site/pages/*.html` holds the site's landing page, Getting Started guide, and Options
Reference — read and edit them directly, no build step needed. `docs_site/assets/style.css`
is their shared stylesheet. `docs_site/pdoc-template/` is a small pdoc template override that
adds a "Docs Home" link back to these pages from the generated API reference; edit it only if
you need to change that link or add another one like it. Run `poe docs` to build the whole
site into `site/`, including the API reference pdoc generates from the docstrings.

## Before publishing a release

- Confirm `[tool.poetry].version` in `pyproject.toml` matches the release tag
  (for example, `0.3.0` for tag `v0.3.0`).
- Publish from a GitHub Release (`published`) or use manual `workflow_dispatch` for a retry (run it on the release tag/commit).

## Reporting bugs or requesting features

Use this repository's [issue tracker](https://github.com/andremun/pyInstanceSpace/issues).

## Security issues

See `SECURITY.md` — please don't report vulnerabilities in a public issue.
