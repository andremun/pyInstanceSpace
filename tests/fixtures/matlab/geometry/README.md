# Controlled MATLAB geometry references

Generated locally on Linux using MATLAB R2026a Update 5 and a clean checkout of
InstanceSpace master `929acfd889e7a17c7ee40c4004ad1bc18639b0e6` (9 October 2026).
The manifest records the runtime, installed toolboxes, source commit, exporter
SHA-256 and each case file hash. `tests/test_matlab_geometry_cases.py` pins the
manifest hash and verifies those file and exporter hashes before comparisons.

To regenerate into a new directory, add `tests/matlab_export` to MATLAB's path and
call `pyis_export_geometry_cases(toolkit_checkout, new_output_directory)`. Review
all numerical changes and update the manifest pin explicitly. The exporter rejects
a dirty source checkout or an existing output directory.

These are small stage-level references with schema
`pyinstancespace.geometry-reference/v1`; they do not replace or approve the full
v2 bundle in `../current`. Candidate/promotion commands for that bundle do not
consume these geometry files.

Cases cover solid, planar, collinear and feature-cap-fallback 3D CLOISTER boundaries,
and 2D TRACE holes and disconnected regions. Comparisons normalize vertex order,
check triangle surface area and convex containment, and compare TRACE membership,
area, topology and counts. Coordinate/area comparisons use 1e-12 absolute tolerance;
counts and membership are exact. Triangulation diagonals and face ordering need not
match MATLAB. MATLAB fixture faces are one-based; Python mesh exports are zero-based.

Python retains the two-array CLOISTER result API. Derived `z_edge_faces` and
`z_ecorr_faces` properties expose triangles for 3D boundaries, including planar
boundaries. CSV export adds `bounds_faces.csv`, `bounds_prunned_faces.csv` and
`bounds_mesh_manifest.json`; indices address rows of the corresponding existing
vertex CSV. Two-dimensional exports remove stale mesh files. TRACE tests cover
boundary CSV round trips and plotting path cycles; no new CLOISTER graph view is
introduced by these references.
