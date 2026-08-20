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

3. **Constrain PYTHIA KNN tuning**
   - Derive the smallest cross-validation training-fold size.
   - Apply it to Sobol and Bayesian KNN parameter generation.
   - Promote invalid-neighbour warnings to errors in focused tests.

4. **Re-audit #272**
   - Reproduce the reported boundary case with the local engine.
   - Compare multi-region topology and membership with R2026a.
   - Fix only a confirmed mismatch; otherwise document the issue as superseded.

5. **Complete #262**
   - Add validated 2D/3D PILOT options and result shapes.
   - Generalize analytic, numerical, and PLS projection paths.
   - Port MATLAB viewpoint grouping/selection.
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
