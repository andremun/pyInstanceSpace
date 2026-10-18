# Fixture provenance and reproduction

## Approved references

`tests/fixtures/matlab/current/` is the installed `matlab-verified`
`reference-export/v2` oracle. Its manifest and verifier profile define the exact
423-file contract. The separate geometry cases have their own manifest and pins;
see [their reproduction guide](../tests/fixtures/matlab/geometry/README.md).

| Approved bundle identity | Value |
|---|---|
| MATLAB source | `2e2cf7565aee8e60190e788c959405d44c4317f3` |
| Python generator identity | `ab72f6bc5c602e0cb03a65041a4677037282e92c` |
| Equivalent generator source in a fresh clone | `41c3df4db2b4adb2b948ea13156faa6ddf5d24c1` |
| Exporter SHA-256 | `ebd7917169ca110fd26856ac8cbad43e3c3477b1ce96473e4f5bc9445ff869d6` |
| Artifact content root | `4c4279651bf9f13b20a5e539cd5b5a123beaeb9ebea4a6699b6d0c1a05bc036a` |
| Runtime | MATLAB R2026a Update 5, Linux GLNXA64 |

The recorded generator and the equivalent revision have identical exporter
contents. The original identity belongs to the approved export: do not rewrite
manifest identities or verifier pins to make them resolve in a fresh clone.
To inspect the equivalent source:

```sh
git show 41c3df4db2b4adb2b948ea13156faa6ddf5d24c1:tests/matlab_export/pyis_export_reference_data.m | sha256sum
```

Exact original generator identity requires its archived source checkout. A new
export from the equivalent revision has a new generator identity and remains a
candidate until reviewed, even when all numerical artifacts match.

## Scientific limits

Two independent MATLAB processes produced identical artifact hashes for the
approved Linux export. TRACE uses explicit boundary tolerance zero and full
coordinate precision. The verifier independently checks schemas, shapes,
resolved options, geometry, topology, membership, metrics and PILOT solver
lineage. Integrity alone does not establish equivalence to newer MATLAB source.

A projection perturbation can change exact boundary membership; legacy
contradiction trimming can also amplify it into different geometry. A positive
distance tolerance does not guarantee continuous legacy trimming. Do not loosen
all test tolerances or silently bless changed fixtures to hide these effects.

Legacy snapshots under `tests/matlab_reference/` have unknown source provenance.
They detect regressions but cannot establish current MATLAB parity. Frozen v1
verified bundles remain readable; they do not replace the installed v2 oracle.

## Export, review and verify

Use the [MATLAB exporter guide](../tests/matlab_export/README.md) for generation,
candidate validation and immutable promotion-review commands. Use the actual
clean MATLAB/Python source identities and an unused output directory. Diagnostic
exports are not oracles. Updating a candidate manifest does not approve it.

Promotion requires reviewed numerical differences, the complete bundle, any
exporter/profile changes, and the corresponding verifier pins and assertions in
one change. Run:

```sh
poetry run python tools/fixture_provenance.py verify tests/fixtures/matlab/current
poetry run python tools/fixture_provenance.py inventory . tests/fixture_inventory.json
poetry run pytest tests/test_fixture_provenance.py tests/test_current_matlab_*.py -W error
```

Keep the full artifact set and independent geometry references intact during
ordinary documentation or dependency work. Their raw data, manifests, source
identities and pin checks remain necessary reproducibility evidence.
