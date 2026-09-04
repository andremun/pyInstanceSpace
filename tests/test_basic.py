"""Package-level smoke tests."""

import instancespace


def test_declared_public_api_is_importable() -> None:
    """Every declared package export must exist and be declared only once."""
    exports = instancespace.__all__
    assert exports, "instancespace must expose a non-empty public API"
    assert len(exports) == len(set(exports)), "instancespace.__all__ has duplicates"

    missing = sorted(name for name in exports if not hasattr(instancespace, name))
    assert not missing, f"instancespace.__all__ contains missing exports: {missing}"
