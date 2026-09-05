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

## Before publishing a release

- Confirm `[tool.poetry].version` in `pyproject.toml` matches the release tag
  (for example, `0.3.0` for tag `v0.3.0`).
- Publish from a GitHub Release (`published`) or use manual `workflow_dispatch` for a retry.

## Reporting bugs or requesting features

Use this repository's [issue tracker](https://github.com/andremun/pyInstanceSpace/issues).

## Security issues

See `SECURITY.md` — please don't report vulnerabilities in a public issue.
