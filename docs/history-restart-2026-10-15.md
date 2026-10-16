# Repository history restart (15 October 2026)

Python `main` now starts with Mario Andrés Muñoz's 15 November 2024 commit.
The migration removed 600 earlier ancestor commits and retained 273 commits on
main, plus seven commits unique to active dependency branches. Trees, author and
committer metadata, dates and messages were checked before publication. Commit
signatures were removed because their signed parent IDs changed.

The repository URL, issues and PR numbers are unchanged. Contributors after the
cutoff remain credited. The new root records the pre-existing snapshot as added
lines under its author; line statistics must not be read as newly written code.

## Locate an earlier Python revision

[history-restart-map.tsv](history-restart-map.tsv) maps the 280 retained original
commit IDs to their rewritten equivalents at migration time. Subsequent commits
are outside this mapping. Dated reports retain their original IDs as historical
evidence; Python GitHub source links use equivalent rewritten revisions when a
mapping exists. MATLAB commit IDs did not change.

| Purpose | Original Python ID | Equivalent rewritten ID |
| --- | --- | --- |
| Approved fixture exporter | `ab72f6bc5c602e0cb03a65041a4677037282e92c` | `41c3df4db2b4adb2b948ea13156faa6ddf5d24c1` |
| Completed docs/dead-code audit merge | `f9eb2d1f35b74edadd2098c4f041432ec4e345fb` | `b07d4209f984d16df2f013fe7a17f9d6bc17be25` |

## Approved fixture provenance

Do not change the installed manifest or verifier pins to the rewritten IDs.
They identify the sources that produced the approved artifacts. The original
exporter revision and its rewritten counterpart have identical file trees.
The exporter script SHA-256 remains
`ebd7917169ca110fd26856ac8cbad43e3c3477b1ce96473e4f5bc9445ff869d6`.

The equivalent source can be inspected in a fresh clone:

```bash
git show 41c3df4db2b4adb2b948ea13156faa6ddf5d24c1:tests/matlab_export/pyis_export_reference_data.m | sha256sum
```

Use a checkout of the original revision from the archived bundle to reproduce
its exact original source identity. An export made from the rewritten revision
has a new generator identity and remains a candidate until reviewed, even when
its source tree and numerical results match.

The maintainer's historical checkout contains the complete pre-rewrite bundles
under `.cache/history-restart-2026-10-11/original-history.bundle` and
`.cache/history-migration-ready/before.bundle`. The latter is the exact remote-ref
snapshot immediately before migration. Preserve these archives independently of
the active checkout. They are local backups, not files supplied by a fresh clone.

## Working copies and releases

Use a fresh clone for development. Do not merge old local branches into rewritten
main; port needed changes onto the new base. The three MATLAB raw-export branches
and superseded development branches were removed after backup. Old tags v0.1.0,
v0.1.1 and v0.2.0 were removed; v0.2.1, v0.2.2 and v0.3.0 point to equivalent
rewritten commits. Existing GitHub releases and PR references can still expose
older history; the restart is not an erasure of all historical objects.
