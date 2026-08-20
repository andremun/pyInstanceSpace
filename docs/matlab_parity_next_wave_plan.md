# MATLAB parity next-wave plan

## Status and scope

This implementation pass runs on `codex/matlab-parity-next-wave`, not `main`. It starts
from `v0.9.0/development-branch-QSF` at `3a7f21a`, which contains branch 1
`codex/open-issue-big-rocks` at `d175048` through PR #319 (`324830c`). It carries branch 2
`codex/validation-serialization-trace3` at `67c73de` through merge `030937d`.

MATLAB InstanceSpace `34c0129` under R2026a Update 4 is the behavioral authority. The
verified 423-file v2 oracle is installed. Implementation and scientific gates are complete.
Full-suite accounting is 988 passed with 63 documented warnings: 987 passed in the sandbox,
and the sole macOS semaphore-permission failure passed 1/1 outside it.

## Delivery status

1. **Integrated and baselined — complete**
   - Proved development, branch 1, and branch 2 ancestry before branch 3 work.
   - Preserved the completed validation, serialization, provenance, and 2D TRACE3 work.

2. **Resolved reviewer reports — complete**
   - Removed #320's genuinely dead polygon-region helper without changing active simplex
     semantics.
   - Resolved #321 with one `casefold()` JSON-key canonicalizer and conflict tests.

3. **Matched PYTHIA KNN and RNG semantics — complete**
   - Retained KNN's nominal 1--25 domain and applied the MATLAB cap at each fit.
   - Preserved requested parameters in reports and eliminated invalid-neighbour score
     warnings.
   - Applied one-based `seed + i` per algorithm and retained its actual splitter.

4. **Disposed of #272 — complete as superseded**
   - Pinned R2026a's valid multi-region all-points contract.
   - Added no single-region retry and recorded the separate MATLAB CSV helper defect.

5. **Completed #262 — complete and verified**
   - Added validated 2D/3D PILOT options and SIFTED dimensionality propagation.
   - Generalized analytic, numerical, and SIMPLS paths, including MATLAB packing, loss,
     restart, fallback, and dtype contracts.
   - Ported and persisted grouped 3D viewpoints.
   - Verified PILOT/viewpoint build and public explore projection with R2026a v2 fixtures.

6. **Re-baselined #304 — audit complete; evidence deferred**
   - Withdrew the old convergence claim because it used rounded `legacy-unknown` metrics.
   - Confirmed MATLAB's expected-improvement-plus/four-seed configuration and retained
     Python's closest supported analogue without changing the shared budget.
   - Fixed the proven per-algorithm RNG boundary.
   - Did not add a Bayesian optimizer trace to v2; repeated-seed candidate/objective traces
     remain required before any convergence or default change.

7. **Completed #265 and native 3D TRACE3 — complete and verified**
   - Added native 3D projection output, plotting, camera use, and a versioned mesh schema.
   - Added tetrahedral TRACE3 construction, membership, metrics, parallel execution, and
     fixed-geometry explore rescoring.
   - Matched R2026a topology, spectra, raw metrics, membership, and rescored summaries.

8. **Closed out provenance and documentation — focused gates complete**
   - Generated and installed verified `reference-export/v2`: 423 files from clean MATLAB
     `34c0129` and Python generator `cf3cde0`.
   - Passed 84 provenance tests and 36 current-gold readers, including 3/3 native 3D TRACE
     readers.
   - Updated implemented fixes, architecture, and the pending backlog.
   - Recorded the platform-split full-suite result: 987 sandbox passes plus the sole
     semaphore-permission test passing 1/1 outside the sandbox, for 988 passed and 63
     documented warnings.

## Test policy

Tests compare scientific invariants rather than unstable simplex row order, optimizer
rotation, or exact-tie choices. Counts, connectivity, orientation, and full-precision 3D
membership are exact. Floating geometry uses documented combined tolerances. No
`legacy-unknown` or diagnostic artifact is promoted to a MATLAB oracle.
