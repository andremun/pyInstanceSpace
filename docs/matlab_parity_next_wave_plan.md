# MATLAB parity next-wave plan

## Scope

This pass starts from the active v0.9 development branch, carries forward the completed
reliability/TRACE3 work, resolves the two new reviewer reports, and then implements the
five prioritized follow-ups. `main` is not an integration source.

## Sequence

1. **Integrate and baseline**
   - Refresh `origin/v0.9.0/development-branch-QSF`.
   - Create `codex/matlab-parity-next-wave` and merge the completed branch.
   - Prove both ancestries and run the full suite before edits.

2. **Resolve reviewer reports**
   - Audit #320/#321 against MATLAB and current Python contracts; record that #322 is not
     present in the public tracker.
   - Remove only genuinely dead alpha-region code; retain simplex semantics.
   - Use one JSON-key canonicalizer and cover Unicode-equivalent conflicts.

3. **Match PYTHIA KNN fitting semantics**
   - Preserve MATLAB's nominal 1--25 Sobol and Bayesian search range.
   - Cap neighbours independently when each fold or final model is fitted, while retaining
     the requested parameter in reports.
   - Remove the incompatible precalculated upper rejection and promote invalid-neighbour
     warnings to errors in focused tests.

4. **Dispose of #272 as superseded**
   - Pin the R2026a two-region all-points contract.
   - Retain valid `MultiPolygon` topology; add no single-region retry.
   - Record the separate MATLAB CSV helper defect for the MATLAB repository.

5. **Complete #262**
   - Add validated 2D/3D PILOT options; make PILOT own SIFTED dimensionality.
   - Generalize analytic, numerical, and PLS projection paths, including MATLAB-order
     solution packing, NaN-loss axes, rank fallback, and double-precision outputs.
   - Port MATLAB viewpoint grouping/selection with `2 x 3` view matrices and radian angles.
   - Add R2026a 3D fixtures and prove existing 2D parity.

6. **Re-baseline #304**
   - Export a verified R2026a Bayesian SVM variant.
   - Compare equal-budget search traces and final models.
   - Correct a proven defect or retain current defaults with an evidence-backed note.

7. **Complete #265 and 3D TRACE3**
   - Define and test the 3D numerical/mesh serialization schema.
   - Add 3D output and visualization without altering 2D schemas.
   - Implement tetrahedral TRACE3 construction, membership, metrics, and rescoring.
   - Verify 3D build and explore against MATLAB.

8. **Close out**
   - Run the full test/static/provenance/parity gates.
   - Update `docs/implemented_fixes.md` and `docs/pending_issue_backlog.md`.
   - Record confirmed divergences and deliberately deferred polish.

## Test policy

Tests compare scientific invariants rather than unstable vertex order, optimizer rotation,
or exact-tie choices. Counts and discrete topology are exact; floating geometry uses
documented tolerances derived from the verified exporter. No legacy-unknown artifact may
be promoted to a MATLAB oracle.
