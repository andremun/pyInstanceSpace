# Reliability and TRACE3 architecture

## Purpose

This pass follows the correctness work on `codex/open-issue-big-rocks`.
It addresses five remaining reliability areas:

1. metadata and option validation;
2. serializer and archive integrity;
3. numerical and warning contracts;
4. MATLAB fixture provenance and layout;
5. a two-dimensional port of MATLAB TRACE3.

The MATLAB repository at commit `34c0129` is the behavioral reference.
Current MATLAB source takes priority over issue prose and old fixtures.
Python-only improvements must preserve the scientific meaning of the pipeline.

## Compatibility classification

| Workstream | Classification | Reason |
|---|---|---|
| Metadata and option rejection | Behavior-changing | Invalid or ambiguous inputs fail before stage execution. |
| Manual selection matching | Behavior-changing | Unknown names no longer select every column. |
| Serializer ownership and safe paths | Corrective | Export cannot mutate a model or escape its output root. |
| Footprint CSV v2 and structured ZIP members | Behavior-changing | Correct geometry and paths replace lossy flat output. |
| Undefined PYTHIA metrics | Behavior-changing | Undefined values become `NaN`, matching MATLAB, rather than emitting a warning and becoming zero. |
| Local alpha-shape primitive | Corrective | Legacy geometry stays stable while deprecated matrix creation is removed. |
| Fixture manifest and verifier | Additive | Existing unknown fixtures are classified, not silently blessed. |
| TRACE3 | Additive | Legacy TRACE remains available and remains Python's default in this pass. |

## Input boundary

### Metadata

Parsed metadata has one case-insensitive `instances` column and at most one `source`
column. Feature and algorithm columns are numeric, non-Boolean, finite or `NaN`, and
unique after their prefixes are removed. Labels, sources, names, and matrix rows and
columns must agree.

A standard build requires at least one instance, three features, and one algorithm.
Feature-only explore data remains valid because trained algorithm labels can supply the
portfolio contract.

Manual feature and algorithm selections accept prefixed or stripped names without case
sensitivity. `None` and an empty list mean "use all." A nonempty request with no matches
is an error. A partial match keeps valid names in dataset order and reports unknown names.

Data washing must reject an empty instance set, fewer than three surviving features, or
no surviving algorithms before calculating ratios or running a stage.

### Options

Every option used by a stage is validated when options are created. Numeric values must
have the expected real, finite, integer, and range properties. Boolean fields reject
integer lookalikes. Arrays must be numeric, finite, two-dimensional, and later match the
dimensions known by their stage.

Cross-field constraints cover SIFTED population sizes, PILOT precalculated arrays,
PYTHIA folds and classifier parameters, and TRACE method-specific behavior. The
`pythia.skip` plus `trace.use_sim` incompatibility remains a legacy TRACE rule. TRACE3
falls back to true labels when PYTHIA is skipped.

## Output boundary

Serialization is a read-only operation on `Model`. Every exported table is created from
a copy. A failed image, CSV, MAT, or archive write raises an exception that names the
operation and target.

Label-derived filenames use deterministic portable stems. Unsafe and colliding labels
remain unchanged in scientific tables and plot titles, but receive unique safe stems on
disk.

Footprint CSV v2 stores one vertex per row with these fields:

| Field | Meaning |
|---|---|
| `Row` | Export row number |
| `Part` | One-based polygon component |
| `Ring` | `exterior` or `hole_N` |
| `Vertex` | One-based vertex within the ring |
| `z_1`, `z_2` | Instance-space coordinates |

The closing coordinate is omitted. Components and holes never share an artificial edge.
Plots use compound paths so holes remain empty.

ZIP output preserves each file's relative path below `output/`. It rejects duplicate
archive members, symlinks, unsafe archive names, and inclusion of the archive itself.

## Numerical boundary

Scaling uses finite-aware min-max helpers. A constant finite slice maps to zero. Missing
values remain missing or masked. All-missing slices do not emit runtime warnings.
Feature and per-algorithm performance columns scale independently. Global performance
uses one range across the complete matrix.

Undefined PYTHIA precision, recall, and selector rates are `NaN`, matching MATLAB.
Selection converts undefined weights to zero before choosing an algorithm. Summary means
and sample standard deviations ignore missing observations without warnings.

Legacy TRACE uses a local two-dimensional alpha-shape primitive based on SciPy Delaunay
triangles and Shapely polygon assembly. This replaces the third-party `np.bmat` path that
produced thousands of `PendingDeprecationWarning` messages. Existing legacy fixtures are
the regression contract for nondegenerate data.

## Fixture trust boundary

MATLAB comparison data is trusted only when a schema-versioned manifest records:

- every file path, SHA-256 digest, shape or schema, role, stage, and variant;
- input files and their hashes;
- clean MATLAB repository commit and toolkit version;
- MATLAB release, platform, and required toolboxes;
- generator repository commit and exporter script hash;
- one linked artifact per variant containing the complete effective MATLAB option tree
  after `ISAvalidateOpts` and `ISAdefaults`, plus the seed, dataset, and timestamp.

Generation uses a fresh temporary directory and publishes atomically. The verifier applies
the strict `reference-export/v1` profile in addition to hashes: it requires both shared
metadata inputs, all three resolved-option artifacts, every base and downstream stage,
build and explore inputs, raw metrics, memberships, and every algorithm footprint. Removing
a file and its manifest entry still fails. Empty geometry is explicit instead of missing.

The reference profile requires MATLAB, Statistics and Machine Learning Toolbox,
Optimization Toolbox, Global Optimization Toolbox, and Financial Toolbox. The last is
required by PRELIM's `boxcox` call and is checked before pipeline execution.

Historical fixtures without this evidence are `legacy-unknown`. Python regression data
is never labeled as a MATLAB oracle.

Issue #278 requires a successful verified run from a clean MATLAB R2025a-or-newer
environment; installed R2026a is the current-gold target. The earlier R2024a diagnostic
completed all variants and supported parity analysis, but predates the complete-option and
explore-input profile, so the hardened verifier does not accept it. Run R2026a only after
both implementations are committed cleanly; review that bundle before installation.

Issue #310 moves only verified data. The canonical installed layout is:

```text
tests/fixtures/matlab/current/
  manifest.json
  shared_inputs/reference/{metadata.csv,metadata_test.csv}
  resolved_options/<variant>.json
  build_data/<stage>/<variant>/{inputs,outputs}/
  explore_data/<stage>/<variant>/{inputs,outputs}/
```

Historical and Python-generated data remain separately classified by
`tests/fixture_inventory.json`; they are not moved into `matlab/current`.

## TRACE3 contract

TRACE3 consumes true `Ybin`, true one-based portfolio `P`, optional PYTHIA predictions
`Yhat`, difficulty labels `beta`, projected coordinates `Z`, and TRACE options. Unlike
legacy TRACE, `use_sim` does not replace truth with predictions. When PYTHIA is available,
its prediction for algorithm `i` filters both the good and best support for `i`. The hard
footprint is never prediction-filtered. TRACE3 does not run contradiction removal.

This pass supports two-dimensional `Z`. Three-dimensional TRACE3 follows PILOT and output
work in issues #262 and #265.

### Alpha-shape engine

The engine uses MATLAB radius units:

1. Delaunay-triangulate unique supporting points.
2. Compute each finite triangle circumradius.
3. Keep triangles whose radius is within the selected alpha radius.
4. Group selected triangles into regions by shared-vertex connectivity, matching MATLAB
   even when polygon interiors touch only at one point.
5. Keep only regions whose combined triangle area is strictly greater than the current
   region threshold.
6. Union the retained triangles while preserving polygon parts and holes.

The default radius models MATLAB `criticalAlpha('all-points')`, not the one-region
criterion stated in issue #313. Degenerate, collinear, or Qhull-failing inputs produce the
canonical empty footprint.

### Footprint construction

For each footprint:

1. Keep unique supporting points. If their count is less than or equal to
   `min_instances`, return empty. The MATLAB default of four therefore needs five points.
2. Build the default alpha shape and calculate raw metrics.
3. Return empty for invalid measure, zero enclosed elements, or measure below
   `min_area_frac` of the full convex-hull measure.
4. Return immediately when purity reaches `purity`.
5. If the alpha spectrum has fewer than two values, return the initial footprint.
6. Otherwise evaluate exactly 100 radii from the initial radius toward the smallest
   spectrum value.
7. Preserve MATLAB's stateful region threshold: apply the previous threshold, set the new
   threshold to the resulting measure divided by 20, then filter again.
8. A later invalid or undersized shape returns empty. The first shape reaching purity is
   returned. If none does, return the last shape, not the best intermediate shape.

Training computes good and best footprints per algorithm, a hard footprint, and full-space
metrics. Explore rescoring keeps trained geometry fixed and recomputes memberships and raw
metrics against new truth. New algorithms receive empty footprints.

### Diagnostic parity evidence

The R2024a diagnostic bundle compared both prediction-filtered TRACE3 and the
PYTHIA-skipped true-label fallback. Every exported good, best, hard, and space result
matched in measure, membership counts, density, purity, components, holes, and boundary
geometry to floating-point precision. Focused probes also confirmed the alpha spectrum,
all-points critical radius, inclusive simplex boundary, strict region threshold, and
shared-vertex region rule. This is implementation evidence only; a clean R2026a run remains
the current-gold provenance gate.

## Default policy

`method="legacy"` remains the Python default until a reviewed current-gold export proves
TRACE3 geometry and metrics. `method="trace3"` is fully usable in two dimensions and tested
by independent invariants plus the R2024a diagnostic comparison.
The method-specific effective purity is 0.55 for omitted legacy options and MATLAB's 0.60
for omitted TRACE3 options. An explicit purity always wins.

The default switch is a separate compatibility decision. It must not be hidden inside the
port.
