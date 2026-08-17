# Open-issue remediation plan

## Baseline

- Repository: `andremun/pyInstanceSpace`
- Source baseline: `origin/v0.9.0/development-branch-QSF` at `e12e92f`
- Work branch: `codex/open-issue-big-rocks`
- MATLAB reference: local `andremun/InstanceSpace` v0.9.0 worktree
- Live tracker: 17 open issues on 2026-08-17

## Scope

This pass fixes verified correctness defects with current user impact.
It does not implement every parity feature.

| Issue | Root cause | Planned correction | Compatibility |
|---|---|---|---|
| #302 | TRACE legacy has broken point masks, empty-state drift, unsupported geometry pieces, and weak edge-case types | Repair all seven verified findings and cover contradiction refinement | Behavior-changing |
| #317 | A positional call reverses accuracy and precision in the PYTHIA summary | Align the signature and use keyword arguments | Behavior-changing |
| #314 | A special branch converts a Boolean prediction into an algorithm index | Use the common weighted-selection formula for all portfolio sizes | Behavior-changing |
| #315 | Explore uses boundary-inclusive TRACE membership | Use boundary-exclusive membership like MATLAB `isinterior` | Behavior-changing |

## TRACE work

The #302 changes must establish one representation for an empty footprint.
`tight()` must return an empty geometry, not `None`.
`Footprint.from_polygon()` must normalize empty geometries to `polygon=None`.

Point membership must use one pointwise helper.
This removes the scalar `contains(MultiPoint)` error from `tight()`.
`tight()` will reuse `fit_poly()` as the boundary-fitting implementation.

Contradictions with no enclosed instances carry no evidence.
The code will keep both footprints and stop that comparison without a warning.

`fit_poly()` will remove triangles with no supporting instances.
`build()` will return the canonical empty footprint when no cluster makes a polygon.
One-dimensional distances will return vectors.
DBSCAN labels will use an integer dtype.

## Test plan

Add focused tests for:

1. TRACE pointwise refinement and empty refinement.
2. Unequal-purity and zero-evidence contradiction paths.
3. Zero-support triangle removal and no-polygon builds.
4. One-dimensional distances and integer DBSCAN labels.
5. PYTHIA single-algorithm selection in build and explore paths.
6. PYTHIA summary columns against raw accuracy and precision.
7. Boundary-exclusive TRACE explore membership.

Run these validation gates after implementation:

1. Targeted PYTHIA and TRACE tests.
2. MATLAB reference tests for PYTHIA and TRACE.
3. The full `poetry run poe test` quality and coverage sequence.
4. A warning-as-error TRACE regression run.

MATLAB is not installed in this workspace.
Issue #278 therefore remains blocked on a real MATLAB export run.

## Deferred work

The current pass does not include TRACE3, 3D PILOT, 3D output, or the stage-contract rewrite.
It also does not reorganize fixtures before provenance is verified.
The final backlog records every deferred open issue and its dependency.
