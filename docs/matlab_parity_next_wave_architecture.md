# MATLAB parity next-wave architecture

## Authority and baseline

- Integration base: `origin/v0.9.0/development-branch-QSF` at `3a7f21a`.
- Continuation branch: `codex/matlab-parity-next-wave`.
- Carried-forward implementation: `codex/validation-serialization-trace3` at
  `67c73de`.
- Gold implementation: MATLAB InstanceSpace at `34c0129`, executed with R2026a.
- Verified 2D reference data: `tests/fixtures/matlab/current/`.

GitHub issues are audit leads, not specifications. A proposed change is accepted only
when it matches MATLAB source or a deliberate, documented Python safety contract.

## Work streams

### Reviewer reports

GitHub exposes issues #320 and #321; #322 currently returns 404. Issue #320 concerns an
unused polygon-region helper beside TRACE3's active simplex-region implementation. The
simplex implementation is authoritative because MATLAB joins alpha-shape simplices that
touch at a vertex. Issue #321 concerns inconsistent normalization of JSON option keys;
both validation and loading must use one canonicalizer.

### PYTHIA tuning constraints

KNN tuning bounds are stage-context data. The largest candidate neighbour count must not
exceed the smallest training fold used by cross-validation. The same runtime bound applies
to Sobol and Bayesian search; public/precalculated parameters keep their existing named
validation. Bounds remain expressed in MATLAB-facing units.

### Alpha-boundary audit

Issue #272 predates the local Delaunay engine. Multi-region geometry is valid and must not
be collapsed into a single polygon. Retry logic is added only for a reproducible mismatch
against current MATLAB; otherwise the issue is documented as superseded.

### Three-dimensional pipeline

PILOT owns the projection dimensionality contract. `dims` is restricted to 2 or 3 and
flows through analytic, numerical, and PLS paths. Existing 2D arrays and numerical results
remain compatible. Three-dimensional PILOT adds viewpoint results without making a
viewpoint part of the fitted projection itself.

After PILOT is stable, output code gains an explicit 3D geometry representation rather
than overloading Shapely polygons. TRACE3 then uses a tetrahedral alpha complex, volume
metrics, point membership, trained-geometry rescoring, and a region-aware mesh export.
Numerical serialization precedes plotting and interactive output.

### Bayesian convergence

Issue #304 is evaluated with a new verified R2026a variant at equal evaluation budgets.
Legacy-unknown CSVs are not optimization oracles. Defaults change only if current evidence
shows a systematic implementation defect; otherwise the documented optimizer difference
remains explicit.

## Compatibility boundaries

- `method="legacy"` remains selectable throughout this pass.
- Any TRACE default switch is a separate versioned decision.
- 3D additions must not change verified 2D outputs outside stated numerical tolerances.
- PRELIM `p` remains one-based; PYTHIA `selection0` remains zero-based with `-1`.
- Serialized 3D geometry receives a versioned schema; 2D footprint CSV v2 remains stable.

## Acceptance evidence

Each work stream requires focused regressions, strict typing, formatting, lint, and a
clean diff. Numerical changes require R2026a source or fixture evidence. Final acceptance
requires the complete Python suite, strict fixture verification and inventory checks,
2D non-regression, new 3D build/explore parity, and concise implemented-fixes and pending
backlog documents.
