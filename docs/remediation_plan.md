# Open-issue remediation plan

> **Superseded historical plan.** Its MATLAB-availability and deferred-work statements
> describe the predecessor `codex/open-issue-big-rocks` baseline. Current status includes
> native 3D PILOT/TRACE3/output and an installed 423-file, `matlab-verified` MATLAB
> R2026a Update 4 v2 oracle. See `docs/architecture.md`, `docs/implemented_fixes.md`,
> and `docs/pending_issue_backlog.md`; the body below is retained as history.

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
| #315 | The report assumes MATLAB excludes polygon-boundary points | Reject the change and pin boundary-inclusive MATLAB parity | No code behavior change |
| Audit | PRELIM `p` is one-based, while PYTHIA `selection0` is zero-based; TRACE and plots treat both alike | Convert only at explicit TRACE and plotting boundaries | Behavior-changing |
| Audit | PRELIM normalizes raw `Y`; validated control flags do not always control execution | Normalize derived performance and honor preprocessing, SIFTED, and parallel flags | Behavior-changing |
| Audit | Stage snapshots alias mutable state and cached models survive reruns | Make rollback transactional and model lifecycle explicit | Behavior-changing |
| Audit | File subsets reject the final valid index and accept invalid indices; early SIFTED exits skip density re-filtering | Validate one-based file indices and centralize density re-filtering | Behavior-changing |

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

## State and preprocessing work

PRELIM will normalize the performance matrix produced by its performance rule.
Its master preprocessing flag will gate both clipping and normalization.
SIFTED will bypass feature selection when disabled and apply density re-filtering
after every enabled selection path.

StageRunner will store independent wave snapshots.
Running until a stage will include that stage.
Every public execution path will finalize or invalidate model state consistently.

The public data contract remains mixed by design: PRELIM `p` is MATLAB-compatible
and one-based; PYTHIA `selection0` is zero-based with `-1` for no selection.
TRACE and plotting code will convert these values only where a common internal
representation is required.

## Test plan

Add focused tests for:

1. TRACE pointwise refinement and empty refinement.
2. Unequal-purity and zero-evidence contradiction paths.
3. Zero-support triangle removal and no-polygon builds.
4. One-dimensional distances and integer DBSCAN labels.
5. PYTHIA single-algorithm selection in build and explore paths.
6. PYTHIA summary columns against raw accuracy and precision.
7. Boundary-inclusive TRACE explore membership.
8. PRELIM derived-performance normalization, NaN handling, and disabled options.
9. SIFTED disabled and density early-exit paths.
10. Inclusive stage execution, isolated rollback, and cached-model invalidation.
11. PRELIM-to-TRACE and serializer index contracts without fixture-side conversion.
12. Valid and invalid one-based subset-file indices.

Run these validation gates after implementation:

1. Targeted PYTHIA and TRACE tests.
2. MATLAB reference tests for PYTHIA and TRACE.
3. The full `poetry run poe test` quality and coverage sequence.
4. A warning-as-error TRACE regression run.

MATLAB is not installed in this workspace.
Issue #278 therefore remains blocked on a real MATLAB export run.

Issue #315 is not implemented. The
[official MATLAB definition](https://www.mathworks.com/help/matlab/ref/polyshape.isinterior.html)
says `polyshape.isinterior` includes boundary points, and the reference fixtures agree.
Changing Python to boundary-exclusive membership reduces fixture agreement.

## Deferred work

The current pass does not include TRACE3, 3D PILOT, 3D output, or a new stage API.
It also does not reorganize fixtures before provenance is verified.
The final backlog records every deferred open issue and its dependency.
