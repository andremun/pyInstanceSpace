# Options Reference

Every supported configuration field, with defaults read from the Python classes

Use immutable option dataclasses or load a nested JSON object. Omit settings you do not need to change.

## Syntax

```python
from instancespace.data.options import InstanceSpaceOptions, PilotOptions

options = InstanceSpaceOptions.default(pilot=PilotOptions.default(dims=3))
options = InstanceSpaceOptions.from_dict({"pilot": {"dims": 3}})
```

## Configuration Rules

The tables use canonical Python field names. JSON accepts these names and the explicit aliases shown, case-insensitively. Validation rejects unknown or conflicting fields. Construct a new configuration with `dataclasses.replace`; do not mutate a frozen dataclass.

Defaults below describe `InstanceSpaceOptions.default()`. A stage seed of `None` inherits `general.seed`. TRACE's omitted purity is method-dependent: 0.55 for `legacy`, 0.60 for `trace3`. Select `method="trace3"` explicitly when that policy is required.

For direct PRELIM calls, `PrelimOptions.from_options(options)` composes `perf`, `bound.flag`, `norm.flag`, `auto.preproc` and `prelim`. It does not add JSON fields. Exact constructor signatures and composed fields are in the [options API](api/instancespace/data/options.html).
