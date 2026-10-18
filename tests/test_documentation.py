# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Mario Andrés Muñoz
"""Executable examples, option coverage and stable documentation boundaries."""

from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from docs_site import build as site
from instancespace.data.options import InstanceSpaceOptions, TraceOptions
from instancespace.instance_space import InstanceSpace
from instancespace.model import Model

# Executing actual published snippets requires their heterogeneous namespace.
# ruff: noqa: ANN401


def snippets(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL)


def test_all_options_have_source_and_website_descriptions() -> None:
    options = InstanceSpaceOptions.default()
    assert site.check_catalog() == 74
    for group in fields(options):
        value = getattr(options, group.name)
        doc = type(value).__doc__ or ""
        for field in fields(value):
            assert re.search(rf"^    {field.name} : .+", doc, re.MULTILINE)
            assert re.sub(r"\s+", "", site.CATALOG[group.name][field.name]) in re.sub(
                r"\s+",
                "",
                doc,
            )
    assert TraceOptions.default(method="legacy").purity == 0.55
    assert TraceOptions.default(method="trace3").purity == 0.60


def test_documented_aliases_are_accepted_by_real_loader() -> None:
    options = InstanceSpaceOptions.default()
    for group, aliases in site.option_aliases().items():
        for alias, name in aliases.items():
            if name == "_":  # Accepted compatibility input, deliberately ignored.
                continue
            original = getattr(getattr(options, group), name)
            loaded = InstanceSpaceOptions.from_dict({group: {alias: original}})
            assert getattr(getattr(loaded, group), name) == original
    assert InstanceSpaceOptions.from_dict({"pilot": {"ISA3D": True}}).pilot.dims == 3
    assert InstanceSpaceOptions.from_dict({"pythia": {"uselibsvm": True}}) == options
    with pytest.raises(ValueError, match="ISA3D"):
        InstanceSpaceOptions.from_dict({"pilot": {"ISA3D": True, "dims": 2}})


def test_builder_preserves_unowned_directories(tmp_path: Path) -> None:
    keep = tmp_path / "important.txt"
    keep.write_text("keep")
    with pytest.raises(ValueError, match="Refusing"):
        site.build(tmp_path)
    assert keep.read_text() == "keep"


def test_api_docstring_anchors_are_scoped() -> None:
    first = site.namespace_docstring('<h2 id="examples">A</h2>', "Model.save")
    second = site.namespace_docstring('<a href="#examples">B</a>', "Model.save")
    assert 'id="Model.save-examples"' in first
    # Only locally defined anchors are renamed; external member links survive.
    assert 'href="#examples"' in second


def test_guides_and_readme_examples_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace: dict[str, Any] = {}
    for code in snippets(site.DOCS / "src/getting-started.md"):
        exec(compile(code, "getting-started.md", "exec", dont_inherit=True), namespace)
    model = namespace["model"]
    assert isinstance(model, Model)
    result = namespace["result"]
    assert result.z.shape == (5, 2)
    assert result.selection0.shape == (5,)
    assert result.y_actual is None
    assert (tmp_path / "output/coordinates.csv").exists()
    restored = InstanceSpace.load("instance-space.checkpoint")
    restored_selection = restored.explore(namespace["new_metadata"]).selection0
    assert restored_selection is not None
    np.testing.assert_array_equal(restored_selection, result.selection0)
    pages = (
        "workflows",
        "model",
        "outputs",
        "preprocessing",
        "prelim",
        "sifted",
        "pilot",
        "pythia",
        "cloister",
        "trace",
        "plugins",
        "tuning",
    )
    for page in pages:
        local = dict(namespace)
        for index, code in enumerate(snippets(site.DOCS / f"src/{page}.md")):
            if page == "model" and index == 0:  # Signature overview, not an example.
                continue
            exec(compile(code, f"{page}.md:{index}", "exec", dont_inherit=True), local)
    namespace["table"].to_csv("metadata.csv", index=False)
    readme = (site.ROOT / "README.md").read_text()
    config = re.search(r"```json\n(.*?)```", readme, re.DOTALL)
    assert config is not None
    Path("options.json").write_text(json.dumps(json.loads(config[1])))
    for code in snippets(site.ROOT / "README.md"):
        exec(compile(code, "README.md", "exec", dont_inherit=True), {})
    namespace["space"].close()
    restored.close()
    plt.close("all")


def test_user_guides_describe_current_behavior() -> None:
    for path in [site.ROOT / "README.md", *(site.DOCS / "src").glob("*.md")]:
        text = path.read_text()
        assert not re.search(
            r"\b(?:TODO|pending|changelog|version history)\b",
            text,
            re.I,
        )
        assert not re.search(r"(?:PR\s*#?\d+|issues?/\d+|pull/\d+)", text)


def test_full_site_preserves_reference_links_and_option_text(tmp_path: Path) -> None:
    output = tmp_path / "site"
    site.build(output)
    assert site.check_links(output) == 42
    pages = site.inventory(output)
    option_page = pages[(output / "options-reference.html").resolve()]
    assert len([name for name in option_page.ids if name.startswith("opts-")]) == 74
    assert "Model.save_to_csv" in " ".join(option_page.words)
    api_page = pages[(output / "api/instancespace/data/options.html").resolve()]
    assert "PythiaOptions" in api_page.ids
    assert "PythiaOptions.default" in api_page.ids
    for path in output.rglob("*.html"):
        if path == output / "api/index.html":
            continue
        text = path.read_text()
        assert 'aria-label="Contents"' in text
        assert 'aria-label="Breadcrumb"' in text
        assert 'id="search"' in text
        assert not re.search(r"function NamedTuple at 0x[0-9a-f]+", text)
