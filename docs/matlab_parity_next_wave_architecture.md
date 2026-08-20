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

MATLAB keeps the nominal KNN search range at 1--25 and caps `NumNeighbors` separately
when each cross-validation fold or final model is fitted. Python must preserve the requested
MATLAB-facing value for Sobol, Bayesian, and precalculated parameters while applying the
same per-fit cap internally. A global bound based on the smallest fold is incorrect because
it changes models fitted on larger folds. PYTHIA also derives each algorithm's folds and
classifier/search randomness from `seed + i`, where `i` is one-based. Python mirrors that
boundary and returns the actual per-algorithm splitters; identical integer seeds do not imply
identical MATLAB/sklearn fold membership because the stratifiers differ.

### Alpha-boundary audit

R2026a disproves issue #272's Python premise. Its default all-points alpha may intentionally
produce multiple regions; on the pinned two-cluster cloud MATLAB and the local Delaunay
engine both use radius `sqrt(0.5)`, retain two regions with area 1, and include all six
points. TRACE3 preserves that topology instead of retrying toward one polygon. Python
geometry, membership, plotting, and CSV v2 already retain every component. The confirmed
defect is limited to MATLAB's legacy `output/scriptcsv.m::traceAlphaBoundary`, which traces
only the first boundary cycle and is not used by Python.

### Three-dimensional pipeline

PILOT owns the projection dimensionality contract. `dims` is restricted to 2 or 3 and
flows into SIFTED and through analytic, numerical, and PLS paths. MATLAB numerical vectors
use column-major packing and its loss averages instances before columns; both conventions
are explicit stage boundaries. Rank-deficient analytic input falls back to the numerical
solver. Existing 2D scientific results remain pinned while MATLAB-backed dtype, summary,
and precalculated-vector defects are corrected.

Three-dimensional PILOT adds viewpoint results without making a viewpoint part of the
fitted projection itself. Each configured zero-based algorithm group yields one `2 x 3`
view matrix and azimuth/elevation in radians. The objective uses MATLAB's soft
`0.2 * abs(dot(unit(v1), unit(v2)))` penalty. The empty group list means one global group;
overlapping groups remain valid. Python retains its existing PLS build/explore centering
asymmetry because an R2026a probe confirms MATLAB has the same behavior.

After PILOT is stable, output code gains an explicit 3D geometry representation rather
than overloading Shapely polygons. TRACE3 then uses a tetrahedral alpha complex, volume
metrics, point membership, trained-geometry rescoring, and a region-aware mesh export.
Numerical serialization precedes plotting and interactive output.

### Bayesian convergence

Issue #304 is evaluated with a new verified R2026a variant at equal evaluation budgets.
Legacy-unknown CSVs are not optimization oracles. Defaults change only if current evidence
shows a systematic implementation defect; otherwise the documented optimizer difference
remains explicit. MATLAB explicitly uses expected-improvement-plus with four seed points;
skopt's plain EI is the closest base acquisition but does not implement MATLAB's additional
anti-overexploitation loop.

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
