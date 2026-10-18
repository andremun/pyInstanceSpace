# pyInstanceSpace contributor context

This repository implements Instance Space Analysis in Python. The package runs
behind a production web service; preserve public wrappers, option names, result
schemas, checkpoints and plugin inputs unless the requested change includes an
explicit compatibility change.

## Current guidance

- Use the repository [ISA skill](.claude/skills/instance-space-analysis/SKILL.md)
  for running, interpreting or advising on an ISA analysis.
- Read [the architecture](docs/architecture.md) for ownership and scientific
  invariants, and [CONTRIBUTING.md](CONTRIBUTING.md) for checks and documentation.
- Current source and tests determine Python behavior. Current MATLAB main/master
  and verified execution are the comparison target; approved fixtures remain
  immutable snapshots of their pinned sources. Papers and issue descriptions
  do not establish what either implementation currently does.
- GitHub Issues are the live backlog. Git records completed plans and audits;
  do not recreate those documents or infer pending work from old phase labels.

## Change and validation contract

Classify PRs as **Additive** or **Behavior-changing**. For behavior changes, state
who sees different results, why, and the before/after scientific verification in
the PR and release notes. Check every consumer of a changed value and avoid
introducing redundant fields or duplicate scientific implementations.

Build logic belongs to stages; fitted inference belongs to `PredictiveStage`.
`InstanceSpace` owns orchestration and metadata reconciliation. Preserve its
compatibility wrappers and the independent build-only plugin contract.

Use the locked Python 3.12 development environment. Run checks appropriate to the
change before reporting completion. The release gate is Ruff, strict mypy, Black,
the checked documentation build, strict pytest with warnings as errors, and the
configured 85% branch-coverage gate. See the workflow and `pyproject.toml` for
exact commands. Record any concrete validation limitation rather than promising
to verify later.

## Documentation and release boundaries

Keep README, `docs_site/src/`, API docstrings and this file focused on current
behavior. Update option descriptions in both `docs_site/options.json` and the
source dataclasses. Preserve API URLs and member anchors. Do not add completed
work logs, branch inventories or duplicated local backlogs.

Keep reproducible numerical evidence and fixture provenance when it supports a
current scientific claim. See [the maintainer documentation](docs/README.md).
Generating fixtures does not approve them; follow the candidate/promotion
workflow. Do not relax tolerances to match rounded output.

A release-preparation task does not itself authorize publication or a MATLAB-side
change. Follow the user's actual scope. Match the repository's PolyForm
Noncommercial source headers and conventional commit style. Use clear, direct
prose; do not require unavailable writing skills.
