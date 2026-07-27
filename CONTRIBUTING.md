# Contributing

Thanks for considering a contribution to `pyInstanceSpace`.

## Getting set up

Follow the *Development Environment Setup Guide* in `README.md` to install Poetry and
the project's dependencies.

## Before opening a PR

- Run `poetry run pytest` and make sure it passes.
- Run `poe test` (ruff, mypy `--strict`, and black format checks — all already
  configured for this project). Note: `poe test` does not currently run `pytest`
  itself (tracked separately); run both commands until that's fixed.
- Code style: ruff for linting, mypy `--strict` for type checking, black for
  formatting — no additional setup needed, just run the commands above.
- Use conventional-commit-style messages (`fix:`, `feat:`, `chore:`, ...), matching
  the existing commit history.
- If your change alters existing behaviour, add an entry to `RELEASE_NOTES.md`.

## Reporting bugs or requesting features

Use this repository's [issue tracker](https://github.com/andremun/pyInstanceSpace/issues).

## Security issues

See `SECURITY.md` — please don't report vulnerabilities in a public issue.
